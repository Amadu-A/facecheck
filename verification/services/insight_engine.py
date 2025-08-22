# verification/services/insight_engine.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis
import onnxruntime as ort

logger = logging.getLogger("app")


# ===================== Конфиг движка =====================

@dataclass
class InsightConfig:
    weights_root: Path
    bundle: str = "buffalo_l"
    use_gpu: bool = True
    det_size: Tuple[int, int] = (960, 960)
    det_thresh: float = 0.30
    threshold: float = 0.60
    doc_angles: Tuple[float, ...] = (0.0, 90.0, -90.0, 15.0, -15.0, 30.0, -30.0)
    selfie_max_side: int = 2000
    doc_max_side: int = 2200
    time_budget_sec: float = 15.0     # тайм-аут на обработку документа
    debug_dir: Optional[Path] = None  # если нужен дамп картинок


# ===================== Утилиты =====================

def _providers(use_gpu: bool) -> List[str]:
    avail = set(ort.get_available_providers())
    if use_gpu and "CUDAExecutionProvider" in avail:
        # Можно попробовать TensorrtExecutionProvider первым, но он иногда долго билдит движок.
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def imread_exif(path: str | Path) -> np.ndarray:
    im = Image.open(str(path))
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def pick_best_face(faces):
    if not faces:
        return None
    # побольше бокс + повыше score
    return sorted(
        faces,
        key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), getattr(f, "det_score", 0.0)),
        reverse=True,
    )[0]


def filter_doc_faces(
    faces,
    img_shape: Tuple[int, int, int],
    min_area_frac: float = 0.003,
    border_frac: float = 0.06,
    min_ar: float = 0.7,
    max_ar: float = 1.6,
    prefer_bottom: bool = True,
):
    H, W = img_shape[:2]
    area_img = W * H
    x_border = int(W * border_frac)
    y_border = int(H * border_frac)

    good = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        ar = w / h

        if area < min_area_frac * area_img:
            continue
        if x1 < x_border or y1 < y_border or x2 > W - x_border or y2 > H - y_border:
            continue
        if not (min_ar <= ar <= max_ar):
            continue

        bonus = 0.15 if (((y1 + y2) / 2) > H * 0.45 and prefer_bottom) else 0.0
        score = ((area / area_img) * 0.7 + getattr(f, "det_score", 0.0) * 0.3) + bonus
        good.append((score, f))

    if not good:
        return faces
    good.sort(key=lambda t: t[0], reverse=True)
    return [f for _, f in good]


def pick_best_doc_face(faces, img_shape):
    if not faces:
        return None
    filtered = filter_doc_faces(faces, img_shape)
    if not filtered:
        return None
    return pick_best_face(filtered)


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ===================== Движок =====================

class InsightEngine:
    def __init__(self, cfg: InsightConfig):
        self.cfg = cfg
        prov = _providers(cfg.use_gpu)
        ctx_id = 0 if "CUDAExecutionProvider" in prov else -1

        # Инициализация FaceAnalysis + единоразовый warm-up
        t0 = time.perf_counter()
        self.app = FaceAnalysis(name=cfg.bundle, root=str(cfg.weights_root), providers=prov)
        self.app.prepare(ctx_id=ctx_id, det_size=cfg.det_size, det_thresh=cfg.det_thresh)

        # Тёплый старт: один короткий вызов на маленьком кадре, чтобы догрузить модели/ядра
        try:
            dummy = np.zeros((480, 640, 3), np.uint8)
            _ = self.app.get(dummy)
        except Exception as e:
            logger.warning("Insight warm-up failed: %s", e)

        logger.info(
            "InsightEngine ready: providers=%s, ctx_id=%s, det=%s (init %.1f ms)",
            prov, ctx_id, cfg.det_size, (time.perf_counter() - t0) * 1000.0
        )

    # ---- Публичный API: обработка пары изображений ----
    def process_pair(self, doc_path: str, selfie_path: str) -> Dict[str, Any]:
        C = self.cfg
        t_all0 = time.perf_counter()

        # 1) чтение с EXIF-ориентацией + даунскейл
        t0 = time.perf_counter()
        doc = imread_exif(doc_path)
        selfie = imread_exif(selfie_path)
        if doc is None or selfie is None:
            return {"ok": False, "reason": "io_error"}
        selfie = resize_max_side(selfie, C.selfie_max_side)
        doc = resize_max_side(doc, C.doc_max_side)
        logger.info("io+exif: %.1f ms | shapes: doc=%s selfie=%s",
                    (time.perf_counter() - t0) * 1000.0, doc.shape, selfie.shape)

        # 2) селфи: детекция 1 раз без поворотов
        t0 = time.perf_counter()
        faces_selfie = self.app.get(selfie)
        best_selfie = pick_best_face(faces_selfie)
        if best_selfie is None:
            return {"ok": False, "reason": "selfie_face_not_found"}
        emb_selfie = best_selfie.normed_embedding
        logger.info("selfie detect+embed: %.1f ms | faces=%d",
                    (time.perf_counter() - t0) * 1000.0, len(faces_selfie) if faces_selfie else 0)

        # 3) документ: перебор углов с тайм-бюджетом
        best_score = -1.0
        best_face = None
        best_angle = 0.0
        start_budget = time.perf_counter()

        # Сначала грубые 3 угла, затем мелкие (ускоряет первые результаты)
        coarse = [a for a in C.doc_angles if abs(a) in (0.0, 90.0)]
        fine = [a for a in C.doc_angles if abs(a) not in (0.0, 90.0)]
        angles = list(dict.fromkeys(coarse + fine))  # сохранить порядок без дублей

        for ang in angles:
            if (time.perf_counter() - start_budget) > C.time_budget_sec:
                logger.warning("doc loop: time budget exceeded (>%ss), stop at angle=%s",
                               C.time_budget_sec, ang)
                break

            t_loop = time.perf_counter()
            img = rotate_img(doc, ang)

            # базовый проход
            faces = self.app.get(img)
            face = pick_best_doc_face(faces, img.shape)

            # осторожный fallback: если не нашли — попробуем снизить фильтры, взять крупнейший детект
            if face is None and faces:
                face = pick_best_face(faces)

            if face is not None:
                s = cosine_sim(face.normed_embedding, emb_selfie)
                if s > best_score:
                    best_score, best_face, best_angle = s, face, ang

            logger.info("doc angle=%s°: faces=%d, best=%.4f (loop %.1f ms)",
                        ang, len(faces) if faces else 0, best_score,
                        (time.perf_counter() - t_loop) * 1000.0)

        if best_face is None:
            return {"ok": False, "reason": "doc_face_not_found"}

        # 4) решение
        verified = bool(best_score >= C.threshold)
        dt_all = (time.perf_counter() - t_all0) * 1000.0
        logger.info("TOTAL: cosine=%.4f thr=%.2f angle=%s° | %.1f ms",
                    best_score, C.threshold, best_angle, dt_all)

        return {
            "ok": True,
            "cosine": float(best_score),
            "verified": verified,
            "best_angle": float(best_angle),
            "elapsed_ms": dt_all,
        }
