# verification/views.py
from __future__ import annotations

import base64
import json
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    JsonResponse,
)
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView, View, FormView
from mlflow import (  # type: ignore
    log_artifacts,
    log_metric,
    set_experiment,
    set_tracking_uri,
    start_run,
)

from mlface_verify.decorators import log_call
from .forms import DocumentUploadForm, SelfieUploadForm
from .services.face_utils import exif_autorotate_bgr
from .services.insight_engine import InsightEngine, InsightConfig
from .services.matcher import match
from .services.face_embedder import FaceEmbedder, FaceEmbedderConfig
from .services.yolo_face_detector import YoloFaceDetector

logger = logging.getLogger("app")


# ============================ МОДЕЛИ ============================

@dataclass
class ModelBundle:
    yolo: Optional[YoloFaceDetector] = None
    embedder: Optional[FaceEmbedder] = None
    insight: Optional[InsightEngine] = None


MODEL_BUNDLE = ModelBundle()


def _ensure_models() -> None:
    """
    Инициализируем модели один раз на процесс.
    По умолчанию используем InsightFace (SCRFD + ArcFace) через ONNXRuntime.
    При отключении — fallback: YOLO + ArcFace (onnx).
    """
    # ---- InsightFace (предпочтительный путь) ----
    use_insight = bool(getattr(settings, "USE_INSIGHTFACE", True))
    if use_insight and MODEL_BUNDLE.insight is None:
        providers = set(ort.get_available_providers())
        cuda_ok = "CUDAExecutionProvider" in providers  # TensorrtExecutionProvider можно добавить по желанию
        cfg = InsightConfig(
            weights_root=Path(getattr(settings, "WEIGHTS_DIR", "weights")),
            bundle=getattr(settings, "INSIGHT_BUNDLE", "buffalo_l"),
            use_gpu=cuda_ok,
            det_size=(960, 960) if cuda_ok else (640, 640),
            det_thresh=0.30,
            threshold=float(getattr(settings, "FACE_MATCH_THRESHOLD", 0.60)),
            doc_angles=(
                0.0, 90.0, -90.0, 15.0, -15.0, 30.0, -30.0
            ) if cuda_ok else (0.0, 90.0, -90.0),
            selfie_max_side=2000 if cuda_ok else 1600,
            doc_max_side=2200 if cuda_ok else 1600,
        )
        MODEL_BUNDLE.insight = InsightEngine(cfg)
        logger.info(
            "InsightEngine initialized | gpu=%s | det=%s | bundle=%s",
            cuda_ok, cfg.det_size, cfg.bundle
        )
        return

    # ---- Fallback: YOLO + ArcFace (onnx) ----
    if MODEL_BUNDLE.yolo is None:
        try:
            MODEL_BUNDLE.yolo = YoloFaceDetector(
                weights_path=getattr(settings, "YOLO_WEIGHTS", "weights/yolo11n-face.pt"),
                device=getattr(settings, "DEVICE", "cpu"),
                conf_th=0.25,
            )
        except Exception as e:
            logger.error(
                "YOLO init failed on %s: %s. Re-init on CPU.",
                getattr(settings, "DEVICE", "cpu"), e
            )
            MODEL_BUNDLE.yolo = YoloFaceDetector(
                weights_path=getattr(settings, "YOLO_WEIGHTS", "weights/yolo11n-face.pt"),
                device="cpu",
                conf_th=0.25,
            )

    if MODEL_BUNDLE.embedder is None:
        MODEL_BUNDLE.embedder = FaceEmbedder(
            FaceEmbedderConfig(
                onnx_path=getattr(settings, "ARCFACE_ONNX", "weights/glintr100.onnx"),
                device=getattr(settings, "DEVICE", "cpu"),
                providers=None,
            )
        )


# ============================ СЕССИЯ ============================

def _session_pop(request: HttpRequest, key: str) -> Optional[str]:
    val = request.session.get(key)
    if key in request.session:
        del request.session[key]
        request.session.modified = True
    return val


def _session_set(request: HttpRequest, key: str, value: Optional[str]) -> None:
    request.session[key] = value
    request.session.modified = True


def _session_get(request: HttpRequest, key: str) -> Optional[str]:
    return request.session.get(key)


# ============================ ВСПОМОГАТЕЛЬНОЕ ============================

def _imread_exif(path: Path) -> np.ndarray:
    """Чтение изображения с учётом EXIF-ориентации (RGB->BGR)."""
    im = Image.open(str(path))
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ============================ VIEWS ============================

class HomeView(TemplateView):
    template_name = "verification/base.html"

    @log_call("HomeView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        # Показать результат анализа только один раз (после редиректа)
        status = _session_pop(request, "verification_status")
        sim = _session_pop(request, "verification_sim")

        doc_rel = _session_get(request, "doc_path")
        selfie_rel = _session_get(request, "selfie_path")

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        if doc_rel and not (media_root / doc_rel).exists():
            doc_rel = None
            _session_set(request, "doc_path", None)
        if selfie_rel and not (media_root / selfie_rel).exists():
            selfie_rel = None
            _session_set(request, "selfie_path", None)

        ctx = {
            "status": status,
            "sim": sim,
            "doc_path": doc_rel,
            "selfie_path": selfie_rel,
        }
        return render(request, self.template_name, ctx)


class UploadDocumentView(FormView):
    form_class = DocumentUploadForm
    template_name = "verification/base.html"

    @log_call("UploadDocumentView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        form = self.get_form()
        if not form.is_valid():
            logger.error("Invalid document upload: %s", form.errors)
            return HttpResponseBadRequest("Invalid form")
        file = form.cleaned_data["document"]
        path = default_storage.save(f"documents/{file.name}", file)
        _session_set(request, "doc_path", path)
        logger.info("Document uploaded -> %s", path)
        return redirect(reverse("home"))


class UploadSelfieView(FormView):
    form_class = SelfieUploadForm
    template_name = "verification/base.html"

    @log_call("UploadSelfieView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        form = self.get_form()
        if not form.is_valid():
            logger.error("Invalid selfie upload: %s", form.errors)
            return HttpResponseBadRequest("Invalid form")
        file = form.cleaned_data["selfie"]
        path = default_storage.save(f"selfies/{file.name}", file)
        _session_set(request, "selfie_path", path)
        logger.info("Selfie uploaded -> %s", path)
        return redirect(reverse("home"))


class CaptureSelfieView(View):
    """Принимает base64 PNG с фронта (камера)."""

    @log_call("CaptureSelfieView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        data_url = request.POST.get("image_data")
        if not data_url or not data_url.startswith("data:image/png;base64,"):
            return HttpResponseBadRequest("No image data")
        b64 = data_url.split(",", 1)[1]
        content = base64.b64decode(b64)
        path = default_storage.save("selfies/camera.png", ContentFile(content))
        _session_set(request, "selfie_path", path)
        logger.info("Selfie captured -> %s", path)
        return redirect(reverse("home"))


class AnalyzeView(View):
    """
    Основной пайплайн: InsightFace (SCRFD + ArcFace) через ONNXRuntime.
    Если отключён — fallback YOLO + ArcFace(onnx).
    """

    @log_call("AnalyzeView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        _ensure_models()

        doc_rel = _session_get(request, "doc_path")
        selfie_rel = _session_get(request, "selfie_path")
        logger.info("Analyze: session paths doc=%s, selfie=%s", doc_rel, selfie_rel)

        if not doc_rel or not selfie_rel:
            return HttpResponseBadRequest("Сначала загрузите документ и селфи")

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        doc_abs = media_root / doc_rel
        selfie_abs = media_root / selfie_rel
        if not doc_abs.exists() or not selfie_abs.exists():
            logger.error(
                "Analyze: file missing: doc_exists=%s selfie_exists=%s",
                doc_abs.exists(), selfie_abs.exists()
            )
            return HttpResponseBadRequest("Файлы не найдены. Перезагрузите их.")

        # ---- Режим InsightFace (рекомендуется) ----
        if getattr(settings, "USE_INSIGHTFACE", True) and MODEL_BUNDLE.insight is not None:
            res = MODEL_BUNDLE.insight.process_pair(str(doc_abs), str(selfie_abs))
            if not res.get("ok"):
                reason = res.get("reason", "unknown")
                human = {
                    "selfie_face_not_found": "Лицо на селфи не найдено",
                    "doc_face_not_found": "Лицо на документе не найдено",
                }.get(reason, "Лицо не найдено")
                request.session["verification_status"] = human
                logger.info("Insight result: %s", human)
                return redirect(reverse("home"))

            sim = float(res["cosine"])
            verified = bool(res["verified"])
            request.session["verification_status"] = (
                "Верификация пройдена ✅" if verified else "Верификация не пройдена ❌"
            )
            request.session["verification_sim"] = f"{sim:.4f}"

            # ---- MLflow (если доступен трекинг-сервер) ----
            if getattr(settings, "MLFLOW_TRACKING_URI", ""):
                try:
                    set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                    set_experiment(getattr(settings, "MLFLOW_EXPERIMENT", "face_verification"))
                    with start_run():
                        log_metric("cosine_sim", sim)
                        log_metric("verified", int(verified))
                        # Можно логировать артефакты, если сохраняете кропы в media/crops
                        crops = media_root / "crops"
                        if crops.exists():
                            log_artifacts(str(crops))
                except Exception as e:
                    logger.error("MLflow logging error: %s", e)

            return redirect(reverse("home"))

        # ---- Fallback: YOLO + ArcFace(onnx) ----
        # Чтение и EXIF-поворот
        img_doc = _imread_exif(doc_abs)
        img_selfie = _imread_exif(selfie_abs)

        det_doc = MODEL_BUNDLE.yolo.detect_best_face(img_doc)  # type: ignore
        if det_doc is None:
            request.session["verification_status"] = "Лицо на документе не найдено"
            return redirect(reverse("home"))
        x1, y1, x2, y2 = det_doc.bbox
        crop_doc = img_doc[y1:y2, x1:x2].copy()

        det_selfie = MODEL_BUNDLE.yolo.detect_best_face(img_selfie)  # type: ignore
        if det_selfie:
            xs1, ys1, xs2, ys2 = det_selfie.bbox
            crop_selfie = img_selfie[ys1:ys2, xs1:xs2].copy()
        else:
            crop_selfie = img_selfie.copy()

        vec_doc = MODEL_BUNDLE.embedder.embed(crop_doc)  # type: ignore
        vec_self = MODEL_BUNDLE.embedder.embed(crop_selfie)  # type: ignore
        res2 = match(vec_doc, vec_self, threshold=float(getattr(settings, "FACE_MATCH_THRESHOLD", 0.60)))
        request.session["verification_status"] = (
            "Верификация пройдена ✅" if res2.verified else "Верификация не пройдена ❌"
        )
        request.session["verification_sim"] = f"{res2.cosine_sim:.4f}"
        return redirect(reverse("home"))


# ============================ ДИАГНОСТИКА / ЛОГИ ============================

@method_decorator(csrf_exempt, name="dispatch")
class ClientLogView(View):
    """
    Принимает JSON: {event: str, details: dict|null, ts: int}
    Пишет в общий лог 'app' на уровне INFO.
    """

    @log_call("ClientLogView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            return HttpResponseBadRequest("invalid json")

        event = str(payload.get("event", ""))
        details = payload.get("details", None)
        ts = payload.get("ts", None)
        logger.info("client_event: %s | details=%s | ts=%s", event, details, ts)
        return HttpResponse(status=204)


class DiagnosticsView(View):
    """
    GET -> JSON с результатами:
      - os/platform
      - наличие /dev/video*
      - lsmod | uvcvideo
      - v4l2-ctl --list-devices (если доступно)
      - принадлежность пользователя к группе video (Linux)
    """

    @log_call("DiagnosticsView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        info: Dict[str, Any] = {"platform": platform.platform(), "system": platform.system()}
        sysname = platform.system().lower()

        if sysname == "linux":
            # /dev/video*
            try:
                video_nodes = sorted(
                    [f for f in os.listdir("/dev") if f.startswith("video")]
                ) if os.path.isdir("/dev") else []
            except Exception:
                video_nodes = []
            info["dev_video"] = [f"/dev/{x}" for x in video_nodes]

            # модуль uvcvideo
            try:
                lsmod = subprocess.run(
                    ["lsmod"], capture_output=True, text=True, timeout=2, check=False
                )
                info["uvcvideo_loaded"] = "uvcvideo" in (lsmod.stdout or "")
                info["lsmod"] = {"rc": lsmod.returncode, "stdout": (lsmod.stdout or "")[:4000]}
            except Exception as e:
                info["lsmod_error"] = str(e)

            # v4l2-ctl
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                try:
                    out = subprocess.run(
                        [v4l2, "--list-devices"], capture_output=True, text=True, timeout=3, check=False
                    )
                    info["v4l2_list_devices"] = out.stdout[:4000]
                except Exception as e:
                    info["v4l2_error"] = str(e)
            else:
                info["v4l2_list_devices"] = "v4l2-ctl not found"

            # группа video
            try:
                import grp
                import pwd
                uid = os.getuid()
                user = pwd.getpwuid(uid).pw_name
                groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
                info["user"] = user
                info["groups"] = groups
                info["in_video_group"] = "video" in groups
            except Exception as e:
                info["groups_error"] = str(e)

        elif sysname == "darwin":
            info["note"] = (
                "macOS: камера обычно доступна без /dev/video*, "
                "проверьте разрешение в System Settings → Privacy & Security → Camera."
            )
        elif sysname == "windows":
            info["note"] = (
                "Windows: проверьте 'Настройки → Конфиденциальность и безопасность → Камера' "
                "и драйверы в Диспетчере устройств."
            )
        else:
            info["note"] = "Неизвестная ОС: базовая проверка недоступна."

        logger.info("diagnostics: %s", info)
        return JsonResponse(info, json_dumps_params={"ensure_ascii": False})


class MLDiagnosticsView(View):
    """Диагностика ML-среды: PyTorch/ONNXRuntime/драйверы."""

    @log_call("MLDiagnosticsView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        info: Dict[str, Any] = {}
        # PyTorch
        try:
            import torch
            info["torch_version"] = getattr(torch, "__version__", None)
            info["torch_cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            info["cuda_device_count"] = torch.cuda.device_count() if info["torch_cuda_available"] else 0
            info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
            info["cudnn_enabled"] = getattr(torch.backends.cudnn, "enabled", None)
        except Exception as e:
            info["torch_error"] = str(e)

        # ONNX Runtime
        try:
            info["onnxruntime_providers"] = ort.get_available_providers()
            info["onnxruntime_version"] = getattr(ort, "__version__", None)
        except Exception as e:
            info["onnxruntime_error"] = str(e)

        # Драйвер / nvidia-smi (если доступно)
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                out = subprocess.check_output([nvidia_smi], text=True, timeout=3)
                info["nvidia_smi"] = out.splitlines()[:20]
            except Exception as e:
                info["nvidia_smi_error"] = str(e)

        logger.info("ml_diagnostics: %s", info)
        return JsonResponse(info, json_dumps_params={"ensure_ascii": False})
