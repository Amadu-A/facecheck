from __future__ import annotations
import json
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

@dataclass
class InsightConfig:
    weights_root: Path
    bundle: str = "buffalo_l"
    use_gpu: bool = True
    det_size: Tuple[int, int] = (640, 640)     # легче, чем 960
    det_thresh: float = 0.30
    threshold: float = 0.60
    doc_angles: Tuple[float, ...] = (0.0, 90.0, -90.0)  # только основные
    selfie_max_side: int = 1024                # даунскейл селфи
    doc_max_side: int = 1600                   # даунскейл документа
    early_stop_margin: float = 0.05            # стоп, если >= threshold+margin
    debug_dir: Optional[Path] = None           # сюда положим артефакты

def _providers(use_gpu: bool) -> List[str]:
    avail = set(ort.get_available_providers())
    if use_gpu and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _imread_exif(path: str | Path) -> np.ndarray:
    im = Image.open(str(path))
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def _pick_best(faces):
    if not faces:
        return None
    return sorted(
        faces,
        key=lambda f: ((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), getattr(f, "det_score", 0.0)),
        reverse=True
    )[0]

def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _draw_face(img: np.ndarray, face, color=(0,255,0)):
    vis = img.copy()
    if face is not None:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        if getattr(face, "kps", None) is not None:
            for (x, y) in face.kps.astype(int):
                cv2.circle(vis, (x, y), 2, (255,0,0), -1)
    return vis

class InsightEngine:
    def __init__(self, cfg: InsightConfig):
        self.cfg = cfg
        prov = _providers(cfg.use_gpu)
        # грузим ТОЛЬКО detection + recognition
        t0 = time.perf_counter()
        self.app = FaceAnalysis(
            name=cfg.bundle,
            root=str(cfg.weights_root),
            providers=prov,
            allowed_modules=["detection", "recognition"],  # 🔑 пропускаем genderage/3d landmarks
        )
        ctx_id = 0 if "CUDAExecutionProvider" in prov else -1
        self.app.prepare(ctx_id=ctx_id, det_size=cfg.det_size, det_thresh=cfg.det_thresh)
        # короткий warmup (на маленьком кадре)
        try:
            _ = self.app.get(np.zeros((384, 512, 3), np.uint8))
        except Exception as e:
            logger.warning("Warmup failed: %s", e)
        logger.info("Insight ready: prov=%s ctx=%s det=%s (%.1f ms)",
                    prov, ctx_id, cfg.det_size, (time.perf_counter()-t0)*1000)

    def process_pair(self, doc_path: str, selfie_path: str) -> Dict[str, Any]:
        C = self.cfg
        t_all = time.perf_counter()
        dbg_dir = None

        # I/O + даунскейл
        doc = _imread_exif(doc_path); selfie = _imread_exif(selfie_path)
        doc = _resize_max(doc, C.doc_max_side)
        selfie = _resize_max(selfie, C.selfie_max_side)

        # селфи
        faces_s = self.app.get(selfie)
        best_s = _pick_best(faces_s)
        if best_s is None:
            return {"ok": False, "reason": "selfie_face_not_found"}
        emb_s = best_s.normed_embedding

        # документ: углы + ранняя остановка
        best = (-1.0, None, 0.0)  # (sim, face, angle)
        for ang in C.doc_angles:
            img = doc if abs(ang) < 1e-3 else _rotate(doc, ang)
            faces = self.app.get(img)
            cand = _pick_best(faces)
            if cand is None:
                continue
            s = _cos(cand.normed_embedding, emb_s)
            if s > best[0]:
                best = (s, cand, ang)
            if s >= (C.threshold + C.early_stop_margin):
                break

        if best[1] is None:
            return {"ok": False, "reason": "doc_face_not_found"}

        # отчёты
        if C.debug_dir:
            dbg_dir = C.debug_dir
            dbg_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg_dir / "selfie_det.jpg"), _draw_face(selfie, best_s))
            # для doc сохраним лучший угол
            img_best = doc if abs(best[2]) < 1e-3 else _rotate(doc, best[2])
            cv2.imwrite(str(dbg_dir / "doc_det.jpg"), _draw_face(img_best, best[1], (0,200,255)))
            with (dbg_dir / "report.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "cosine": float(best[0]),
                    "threshold": C.threshold,
                    "best_angle": float(best[2]),
                    "bundle": C.bundle,
                    "det_size": list(C.det_size),
                }, f, ensure_ascii=False, indent=2)

        return {
            "ok": True,
            "cosine": float(best[0]),
            "verified": bool(best[0] >= C.threshold),
            "best_angle": float(best[2]),
            "debug_dir": str(dbg_dir) if dbg_dir else None,
            "elapsed_ms": (time.perf_counter()-t_all)*1000.0,
        }

def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
