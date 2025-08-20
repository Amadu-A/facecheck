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
from typing import Optional, Dict, Any, List

from django.conf import settings
from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
    HttpResponseBadRequest,
)
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import TemplateView, View, FormView
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from PIL import Image
import numpy as np
import cv2

from mlflow import start_run, log_metric, log_artifacts, set_tracking_uri, set_experiment  # type: ignore
from mlface_verify.decorators import log_call
from .forms import DocumentUploadForm, SelfieUploadForm
from .services.image_io import imread, imwrite
from .services.yolo_face_detector import YoloFaceDetector
from .services.face_embedder import FaceEmbedder, FaceEmbedderConfig
from .services.matcher import match
from .services.quality import image_quality_metrics
from .services.face_utils import exif_autorotate_bgr, rotate90k
from .services.debug_vis import draw_box

logger = logging.getLogger("app")

# ——— Инициализация моделей (держим лениво, чтобы старт был быстрым) ———
@dataclass
class ModelBundle:
    yolo: Optional[YoloFaceDetector] = None
    embedder: Optional[FaceEmbedder] = None

MODEL_BUNDLE = ModelBundle()

def _ensure_models():
    if MODEL_BUNDLE.yolo is None:
        try:
            MODEL_BUNDLE.yolo = YoloFaceDetector(
                weights_path=settings.YOLO_WEIGHTS,
                device=settings.DEVICE,
                conf_th=0.25,
            )
        except Exception as e:
            logger.error(f"YOLO init failed on {settings.DEVICE}: {e}. Re-init on CPU.")
            MODEL_BUNDLE.yolo = YoloFaceDetector(
                weights_path=settings.YOLO_WEIGHTS,
                device="cpu",
                conf_th=0.25,
            )
    if MODEL_BUNDLE.embedder is None:
        MODEL_BUNDLE.embedder = FaceEmbedder(
            FaceEmbedderConfig(
                onnx_path=settings.ARCFACE_ONNX,
                device=settings.DEVICE,
                providers=None
            )
        )


# ——— Вспомогательные функции ———
def _session_set(request: HttpRequest, key: str, value: str) -> None:
    request.session[key] = value
    request.session.modified = True

def _session_get(request: HttpRequest, key: str) -> Optional[str]:
    return request.session.get(key)

# ——— Views ———
class HomeView(TemplateView):
    template_name = "verification/base.html"

    @log_call("HomeView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        ctx = {
            "status": request.session.get("verification_status"),
            "sim": request.session.get("verification_sim"),
            "doc_path": _session_get(request, "doc_path"),
            "selfie_path": _session_get(request, "selfie_path"),
        }
        return render(request, self.template_name, ctx)

class UploadDocumentView(FormView):
    form_class = DocumentUploadForm
    template_name = "verification/base.html"

    @log_call("UploadDocumentView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        form = self.get_form()
        if not form.is_valid():
            logger.error(f"Invalid document upload: {form.errors}")
            return HttpResponseBadRequest("Invalid form")
        file = form.cleaned_data["document"]
        path = default_storage.save(f"documents/{file.name}", file)
        _session_set(request, "doc_path", path)
        logger.info(f"Document uploaded -> {path}")
        return redirect(reverse("home"))

class UploadSelfieView(FormView):
    form_class = SelfieUploadForm
    template_name = "verification/base.html"

    @log_call("UploadSelfieView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        form = self.get_form()
        if not form.is_valid():
            logger.error(f"Invalid selfie upload: {form.errors}")
            return HttpResponseBadRequest("Invalid form")
        file = form.cleaned_data["selfie"]
        path = default_storage.save(f"selfies/{file.name}", file)
        _session_set(request, "selfie_path", path)
        logger.info(f"Selfie uploaded -> {path}")
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
        logger.info(f"Selfie captured -> {path}")
        return redirect(reverse("home"))

class AnalyzeView(View):
    """Основной пайплайн анализа: детект лица на документе -> эмбеддинги -> матч -> результат."""

    @log_call("AnalyzeView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        _ensure_models()

        doc_rel = _session_get(request, "doc_path")
        selfie_rel = _session_get(request, "selfie_path")
        logger.info(f"Analyze: session paths doc={doc_rel}, selfie={selfie_rel}")

        if not doc_rel or not selfie_rel:
            logger.error("Analyze: missing doc or selfie")
            return HttpResponseBadRequest("Сначала загрузите документ и селфи")

        doc_abs = settings.MEDIA_ROOT / doc_rel
        selfie_abs = settings.MEDIA_ROOT / selfie_rel
        if not doc_abs.exists() or not selfie_abs.exists():
            logger.error(f"Analyze: file missing: doc={doc_abs.exists()} selfie={selfie_abs.exists()}")
            return HttpResponseBadRequest("Файлы не найдены. Перезагрузите их.")

        # ----- чтение + EXIF-автоповорот -----
        img_doc = cv2.imread(str(doc_abs), cv2.IMREAD_COLOR)
        img_selfie = cv2.imread(str(selfie_abs), cv2.IMREAD_COLOR)
        try:
            img_doc = exif_autorotate_bgr(img_doc, Image.open(str(doc_abs)))
        except Exception:
            img_doc = exif_autorotate_bgr(img_doc)
        try:
            img_selfie = exif_autorotate_bgr(img_selfie, Image.open(str(selfie_abs)))
        except Exception:
            img_selfie = exif_autorotate_bgr(img_selfie)

        logger.info(f"orig doc: {image_quality_metrics(img_doc)}")
        logger.info(f"orig selfie: {image_quality_metrics(img_selfie)}")

        # ----- 1) Детект лица на документе -----
        det_doc = MODEL_BUNDLE.yolo.detect_best_face(img_doc)  # type: ignore
        if det_doc is None:
            request.session["verification_status"] = "Лицо на документе не найдено"
            return redirect(reverse("home"))

        x1, y1, x2, y2 = det_doc.bbox
        crop_doc = img_doc[y1:y2, x1:x2].copy()
        if settings.DEBUG_ANALYSIS:
            dbg_doc = draw_box(img_doc, det_doc.bbox, (0, 200, 255), f"conf={det_doc.conf:.3f}")
            (settings.DEBUG_DIR).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(settings.DEBUG_DIR / "doc_annotated.png"), dbg_doc)

        logger.info(f"doc crop: {image_quality_metrics(crop_doc)} conf={det_doc.conf:.3f} bbox={det_doc.bbox}")

        # ----- 2) Детект лица на селфи -----
        det_selfie = MODEL_BUNDLE.yolo.detect_best_face(img_selfie)  # type: ignore
        if det_selfie:
            xs1, ys1, xs2, ys2 = det_selfie.bbox
            crop_selfie = img_selfie[ys1:ys2, xs1:xs2].copy()
            logger.info(
                f"selfie crop: {image_quality_metrics(crop_selfie)} conf={det_selfie.conf:.3f} bbox={det_selfie.bbox}")
            if settings.DEBUG_ANALYSIS:
                dbg_selfie = draw_box(img_selfie, det_selfie.bbox, (0, 255, 0), f"conf={det_selfie.conf:.3f}")
                cv2.imwrite(str(settings.DEBUG_DIR / "selfie_annotated.png"), dbg_selfie)
        else:
            crop_selfie = img_selfie.copy()
            logger.info("selfie face not found, using full image")
            logger.info(f"selfie full: {image_quality_metrics(crop_selfie)}")

        # ----- 3) Эмбеддинг селфи (один раз) -----
        vec_self = MODEL_BUNDLE.embedder.embed(crop_selfie)  # type: ignore

        # ----- 4) Перебор поворотов для документа -----
        sims = []
        best = (-1.0, 0, crop_doc)  # (sim, k, image)
        for k in (0, 1, 2, 3):  # 0/90/180/270
            rot = rotate90k(crop_doc, k)
            vec_doc = MODEL_BUNDLE.embedder.embed(rot)  # type: ignore
            sim = float(np.dot(vec_doc, vec_self))
            sims.append((k, sim))
            if sim > best[0]:
                best = (sim, k, rot)
            if settings.DEBUG_ANALYSIS:
                cv2.imwrite(str(settings.DEBUG_DIR / f"doc_rot_{k * 90}.png"), rot)

        best_sim, best_k, best_img = best
        cv2.imwrite(str(settings.MEDIA_ROOT / "crops/doc_face_best.png"), best_img)
        cv2.imwrite(str(settings.MEDIA_ROOT / "crops/selfie_face.png"), crop_selfie)
        logger.info(f"rot sims: {[(k, round(s, 4)) for k, s in sims]} -> best={best_k * 90}deg sim={best_sim:.4f}")

        # ----- 5) Решение -----
        verified = best_sim >= settings.FACE_MATCH_THRESHOLD
        status_text = "Верификация пройдена ✅" if verified else "Верификация не пройдена ❌"
        request.session["verification_status"] = status_text
        request.session["verification_sim"] = f"{best_sim:.4f}"

        # ----- 6) MLflow (только если сервер доступен) -----
        if settings.MLFLOW_TRACKING_URI:
            try:
                import requests
                ping = requests.get(settings.MLFLOW_TRACKING_URI, timeout=0.7)  # быстрая проверка
                if ping.ok:
                    set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                    set_experiment(settings.MLFLOW_EXPERIMENT)
                    with start_run():
                        log_metric("doc_face_conf", float(det_doc.conf))
                        if det_selfie:
                            log_metric("selfie_face_conf", float(det_selfie.conf))
                        log_metric("cosine_sim", float(best_sim))
                        log_metric("verified", int(verified))
                        log_artifacts(str(settings.MEDIA_ROOT / "crops"))
                else:
                    logger.error(f"MLflow ping failed: {ping.status_code}")
            except Exception as e:
                logger.error(f"MLflow logging skipped: {e}")

        return redirect(reverse("home"))


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
        logger.info(f"client_event: {event} | details={details} | ts={ts}")
        return HttpResponse(status=204)

# === серверная диагностика камеры (Linux/прочие ОС) ==========================
def _run(cmd: List[str], timeout: int = 3) -> Dict[str, Any]:
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, text=True
        )
        return {"cmd": cmd, "rc": out.returncode, "stdout": out.stdout[:4000], "stderr": out.stderr[:4000]}
    except Exception as e:
        return {"cmd": cmd, "error": str(e)}

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
            video_nodes = sorted([f for f in os.listdir("/dev") if f.startswith("video")]) if os.path.isdir("/dev") else []
            info["dev_video"] = [f"/dev/{x}" for x in video_nodes]

            # модуль uvcvideo
            lsmod = _run(["/sbin/lsmod"]) if os.path.exists("/sbin/lsmod") else _run(["lsmod"])
            info["lsmod"] = lsmod
            info["uvcvideo_loaded"] = "uvcvideo" in (lsmod.get("stdout") or "")

            # v4l2-ctl
            v4l2 = shutil.which("v4l2-ctl")
            if v4l2:
                info["v4l2_list_devices"] = _run([v4l2, "--list-devices"])
            else:
                info["v4l2_list_devices"] = {"error": "v4l2-ctl not found"}

            # группа video
            try:
                import grp, pwd
                uid = os.getuid()
                user = pwd.getpwuid(uid).pw_name
                groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
                info["user"] = user
                info["groups"] = groups
                info["in_video_group"] = "video" in groups
            except Exception as e:
                info["groups_error"] = str(e)

        elif sysname == "darwin":
            info["note"] = "macOS: камера обычно доступна без /dev/video*, проверьте разрешение в System Settings → Privacy & Security → Camera."
        elif sysname == "windows":
            info["note"] = "Windows: проверьте 'Настройки → Конфиденциальность и безопасность → Камера' и драйверы в Диспетчере устройств."
        else:
            info["note"] = "Неизвестная ОС: базовая проверка недоступна."

        logger.info(f"diagnostics: {info}")
        return JsonResponse(info, json_dumps_params={"ensure_ascii": False})


class MLDiagnosticsView(View):
    @log_call("MLDiagnosticsView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        info = {}
        # PyTorch
        try:
            import torch
            info["torch_cuda_available"] = bool(torch.cuda.is_available())
            info["torch_version"] = getattr(torch, "__version__", None)
            info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
            info["cudnn_enabled"] = getattr(torch.backends.cudnn, "enabled", None)
            info["cudnn_version"] = getattr(torch.backends.cudnn, "version", None)
            info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception as e:
            info["torch_error"] = str(e)

        # ONNX Runtime
        try:
            import onnxruntime as ort
            info["onnxruntime_providers"] = ort.get_available_providers()
        except Exception as e:
            info["onnxruntime_error"] = str(e)

        # Библиотеки/среда
        import os, shutil, glob, subprocess
        info["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
        info["CUDA_PATH"] = os.environ.get("CUDA_PATH")
        info["cudnn_libs"] = [p for p in glob.glob("/usr/local/cuda/lib64/libcudnn*")][:10]
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                out = subprocess.check_output([nvidia_smi], text=True, timeout=3)
                info["nvidia_smi"] = out.splitlines()[:20]
            except Exception as e:
                info["nvidia_smi_error"] = str(e)

        logger.info(f"ml_diagnostics: {info}")
        return JsonResponse(info, json_dumps_params={"ensure_ascii": False})
