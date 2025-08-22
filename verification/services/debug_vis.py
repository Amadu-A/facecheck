# verification/services/debug_vis.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple

def draw_box(img: np.ndarray, bbox: Tuple[int, int, int, int], color=(0, 255, 0), text: str | None = None):
    out = img.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(out, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
