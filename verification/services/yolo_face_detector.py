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
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float

class YoloFaceDetector:
    """
    Детектор лица на основе YOLOv11 face.
    Ожидается, что вес обучен на классе "face" (id=0).
    """
    def __init__(self, weights_path: str, device: str = "auto", conf_th: float = 0.25):
        self.model = YOLO(weights_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.conf_th = conf_th
        logger.info(f"YOLO loaded: {weights_path} on {device}")

    def detect_best_face(self, img_bgr: np.ndarray) -> Optional[DetectedFace]:
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, conf=self.conf_th, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            logger.info("YOLO: no faces")
            return None

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().tolist()
        # Выберем лучший bbox: по conf, с легким бонусом за площадь
        best = None
        best_score = -1.0
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            area = (x2 - x1) * (y2 - y1)
            score = float(c) + 0.000001 * area
            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, float(c))

        if best is None:
            return None
        x1, y1, x2, y2, conf = best
        # клиппинг
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        return DetectedFace((x1, y1, x2, y2), conf)
