# verification/services/quality.py
import cv2
import numpy as np
from typing import Dict

def image_quality_metrics(img_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    h, w = gray.shape[:2]
    return {"h": h, "w": w, "brightness": bright, "sharpness": sharp}
