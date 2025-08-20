# verification/services/yolo_face_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("app")

@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    conf: float

class YoloFaceDetector:
    def __init__(self, weights_path: str, device: str = "auto", conf_th: float = 0.25):
        self.model = YOLO(weights_path)
        chosen = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        def _try_device(dev: str) -> bool:
            try:
                self.model.to(dev)
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, conf=0.9, verbose=False)  # прогреваем cudnn
                return True
            except Exception as e:
                logger.error(f"YOLO warmup failed on {dev}: {e}")
                return False

        if not _try_device(chosen):
            logger.info("Falling back to CPU for YOLO due to CUDA/cuDNN issue")
            self.model.to("cpu")
            chosen = "cpu"

        self.conf_th = conf_th
        logger.info(f"YOLO loaded: {weights_path} on {chosen}")

    def detect_best_face(self, img_bgr: np.ndarray) -> Optional[DetectedFace]:
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, conf=self.conf_th, verbose=False)[0]
        if not res.boxes or len(res.boxes) == 0:
            logger.info("YOLO: no faces")
            return None
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy().tolist()
        best, best_score = None, -1.0
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            score = float(c) + 1e-6 * (x2 - x1) * (y2 - y1)
            if score > best_score:
                best_score, best = score, (x1, y1, x2, y2, float(c))
        if not best:
            return None
        x1, y1, x2, y2, conf = best
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        return DetectedFace((x1, y1, x2, y2), conf)
