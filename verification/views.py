# verification/views.py
from __future__ import annotations
import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import TemplateView, View, FormView
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import numpy as np
import cv2

from mlflow import start_run, log_metric, log_artifacts, set_tracking_uri, set_experiment  # type: ignore

from mlface_verify.decorators import log_call
from .forms import DocumentUploadForm, SelfieUploadForm
from .services.image_io import imread, imwrite
from .services.yolo_face_detector import YoloFaceDetector
from .services.face_embedder import FaceEmbedder, FaceEmbedderConfig
from .services.matcher import match

logger = logging.getLogger("app")

# ——— Инициализация моделей (держим лениво, чтобы старт был быстрым) ———
@dataclass
class ModelBundle:
    yolo: Optional[YoloFaceDetector] = None
    embedder: Optional[FaceEmbedder] = None

MODEL_BUNDLE = ModelBundle()

def _ensure_models():
    if MODEL_BUNDLE.yolo is None:
        MODEL_BUNDLE.yolo = YoloFaceDetector(
            weights_path=settings.YOLO_WEIGHTS,
            device=settings.DEVICE,
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
        if not doc_rel or not selfie_rel:
            logger.error("Analyze: missing doc or selfie")
            return HttpResponseBadRequest("Сначала загрузите документ и селфи")

        doc_abs = settings.MEDIA_ROOT / doc_rel
        selfie_abs = settings.MEDIA_ROOT / selfie_rel

        img_doc = imread(str(doc_abs))
        img_selfie = imread(str(selfie_abs))

        # 1) Детект лица на документе
        det_doc = MODEL_BUNDLE.yolo.detect_best_face(img_doc)  # type: ignore
        if det_doc is None:
            request.session["verification_status"] = "Лицо на документе не найдено"
            return redirect(reverse("home"))

        x1, y1, x2, y2 = det_doc.bbox
        crop_doc = img_doc[y1:y2, x1:x2].copy()
        imwrite(str(settings.MEDIA_ROOT / "crops/doc_face.png"), crop_doc)
        logger.info(f"doc face conf={det_doc.conf:.3f} bbox={det_doc.bbox}")

        # 2) Детект лица на селфи (улучшаем устойчивость)
        det_selfie = MODEL_BUNDLE.yolo.detect_best_face(img_selfie)  # type: ignore
        if det_selfie:
            xs1, ys1, xs2, ys2 = det_selfie.bbox
            crop_selfie = img_selfie[ys1:ys2, xs1:xs2].copy()
            logger.info(f"selfie face conf={det_selfie.conf:.3f} bbox={det_selfie.bbox}")
        else:
            # если не нашли — попробуем целиком
            crop_selfie = img_selfie.copy()
            logger.info("selfie face not found, using full image")

        imwrite(str(settings.MEDIA_ROOT / "crops/selfie_face.png"), crop_selfie)

        # 3) Эмбеддинги
        vec_doc = MODEL_BUNDLE.embedder.embed(crop_doc)   # type: ignore
        vec_self = MODEL_BUNDLE.embedder.embed(crop_selfie)  # type: ignore

        # 4) Матч
        result = match(vec_doc, vec_self, threshold=settings.FACE_MATCH_THRESHOLD)
        status_text = "Верификация пройдена ✅" if result.verified else "Верификация не пройдена ❌"
        request.session["verification_status"] = status_text
        request.session["verification_sim"] = f"{result.cosine_sim:.4f}"

        # 5) MLflow (если задан)
        if settings.MLFLOW_TRACKING_URI:
            try:
                set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                set_experiment(settings.MLFLOW_EXPERIMENT)
                with start_run():
                    log_metric("doc_face_conf", float(det_doc.conf))
                    if det_selfie:
                        log_metric("selfie_face_conf", float(det_selfie.conf))
                    log_metric("cosine_sim", float(result.cosine_sim))
                    log_metric("verified", int(result.verified))
                    log_artifacts(str(settings.MEDIA_ROOT / "crops"))
            except Exception as e:
                logger.error(f"MLflow logging error: {e}")

        return redirect(reverse("home"))
