# verification/services/debug_vis.py
from typing import Tuple, List
import cv2
import numpy as np

def draw_box(img: np.ndarray, box: Tuple[int, int, int, int], color=(0,255,0), text: str = "") -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(out, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
