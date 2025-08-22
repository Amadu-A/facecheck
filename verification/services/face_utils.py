# verification/services/face_utils.py
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageOps

def exif_autorotate_bgr(img_bgr, pil: Image.Image | None = None):
    """Автоповорот по EXIF: если есть PIL — корректно учтём, иначе — вернём как есть."""
    if pil is None:
        return img_bgr
    pil = ImageOps.exif_transpose(pil.convert("RGB"))
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def rotate90k(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0: return img
    if k == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2: return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
