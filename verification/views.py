# verification/views.py
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    JsonResponse,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView, View, FormView

from mlface_verify.decorators import log_call
from .forms import DocumentUploadForm, SelfieUploadForm
from .services.insight_engine import InsightEngine, InsightConfig

logger = logging.getLogger("app")


# ============================= Models bundle ==============================

@dataclass
class ModelBundle:
    insight: Optional[InsightEngine] = None


MODEL_BUNDLE = ModelBundle()


def _ensure_engine() -> None:
    """Ленивая инициализация InsightFace-движка (SCRFD + recognition)."""
    if MODEL_BUNDLE.insight is not None:
        return

    cfg = InsightConfig(
        weights_root=Path(settings.WEIGHTS_DIR),
        bundle=getattr(settings, "INSIGHT_BUNDLE", "buffalo_l"),
        use_gpu=True,                 # провайдеры/наличие CUDA проверяются внутри движка
        det_size=(640, 640),
        det_thresh=0.30,
        threshold=float(settings.FACE_MATCH_THRESHOLD),
        doc_angles=(0.0, 90.0, -90.0),
        selfie_max_side=1024,
        doc_max_side=1600,
        early_stop_margin=0.05,
        debug_dir=None,               # подставляем per-request
    )
    MODEL_BUNDLE.insight = InsightEngine(cfg)


# =============================== Session utils ============================

def _session_set(request: HttpRequest, key: str, value: Any) -> None:
    request.session[key] = value
    request.session.modified = True


def _session_get(request: HttpRequest, key: str) -> Optional[Any]:
    return request.session.get(key)


def _session_pop(request: HttpRequest, key: str) -> Optional[Any]:
    val = request.session.get(key)
    if key in request.session:
        del request.session[key]
        request.session.modified = True
    return val


def _time_stamp_dir() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ================================= Views ==================================

class HomeView(TemplateView):
    template_name = "verification/base.html"

    @log_call("HomeView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        # Показываем результат один раз, потом очищаем (по требованию ТЗ)
        ctx = {
            "status": _session_pop(request, "verification_status"),
            "sim": _session_pop(request, "verification_sim"),
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
    """Принимает base64 PNG из камеры (фронтенд)."""

    @log_call("CaptureSelfieView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        data_url = request.POST.get("image_data")
        if not data_url or not data_url.startswith("data:image/png;base64,"):
            return HttpResponseBadRequest("No image data")
        content = base64.b64decode(data_url.split(",", 1)[1])
        path = default_storage.save("selfies/camera.png", ContentFile(content))
        _session_set(request, "selfie_path", path)
        logger.info("Selfie captured -> %s", path)
        return redirect(reverse("home"))


class AnalyzeView(View):
    """Быстрый пайплайн: InsightFace (детекция+эмбеддинг) + косинус, отчёты."""

    @log_call("AnalyzeView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        _ensure_engine()

        doc_rel = _session_get(request, "doc_path")
        selfie_rel = _session_get(request, "selfie_path")
        if not doc_rel or not selfie_rel:
            return HttpResponseBadRequest("Сначала загрузите документ и селфи")

        media_root = Path(settings.MEDIA_ROOT)
        doc_abs = media_root / str(doc_rel)
        selfie_abs = media_root / str(selfie_rel)
        if not doc_abs.exists() or not selfie_abs.exists():
            return HttpResponseBadRequest("Файлы не найдены. Перезагрузите их.")

        # per-request debug директория (как в CLI out_dir)
        debug_dir = media_root / "debug" / _time_stamp_dir()
        MODEL_BUNDLE.insight.cfg.debug_dir = debug_dir  # type: ignore[attr-defined]

        res = MODEL_BUNDLE.insight.process_pair(str(doc_abs), str(selfie_abs))  # type: ignore[union-attr]
        if not res.get("ok"):
            _session_set(request, "verification_status", "Лицо не найдено ❌")
            _session_set(request, "verification_sim", None)
            logger.info("Analyze fail: %s", res)
            return redirect(reverse("home"))

        sim = float(res["cosine"])
        verified = bool(res["verified"])
        _session_set(request, "verification_status", "Верификация пройдена ✅" if verified else "Верификация не пройдена ❌")
        _session_set(request, "verification_sim", f"{sim:.4f}")
        logger.info("Analyze ok: cosine=%.4f verified=%s debug=%s",
                    sim, verified, res.get("debug_dir"))
        return redirect(reverse("home"))


# ============================== Lightweight stubs =========================
# Чтобы не править urls.py и фронт, возвращаем быстрый JSON.

@method_decorator(csrf_exempt, name="dispatch")
class ClientLogView(View):
    """Принимает JSON-события с фронта: POST /client-log/"""

    @log_call("ClientLogView.post")
    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            return HttpResponseBadRequest("invalid json")
        logger.info(
            "client_event: %s | details=%s | ts=%s",
            payload.get("event"),
            payload.get("details"),
            payload.get("ts"),
        )
        return HttpResponse(status=204)


class DiagnosticsView(View):
    """Заглушка для /diagnostics/ — ничего тяжёлого, просто 200 OK."""

    @log_call("DiagnosticsView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        return JsonResponse(
            {
                "status": "ok",
                "note": "Diagnostics disabled (stub).",
            },
            json_dumps_params={"ensure_ascii": False},
        )


class MLDiagnosticsView(View):
    """Заглушка для /ml-diagnostics/ — чтобы не менять маршруты."""

    @log_call("MLDiagnosticsView.get")
    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        return JsonResponse(
            {
                "status": "ok",
                "note": "ML diagnostics disabled (stub).",
            },
            json_dumps_params={"ensure_ascii": False},
        )
