# verification/services/face_utils.py
from typing import Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image, ExifTags

def exif_autorotate_bgr(img_bgr: np.ndarray, pil_image: Optional[Image.Image] = None) -> np.ndarray:
    """
    Если есть EXIF Orientation — приводим к нормальному виду.
    Можно передать готовый PIL Image (если читали файл PIL'ом), иначе сконвертим.
    """
    try:
        if pil_image is None:
            pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        exif = getattr(pil_image, "_getexif", lambda: None)()
        if not exif:
            return img_bgr
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == 'Orientation':
                orientation_key = k
                break
        if orientation_key is None or orientation_key not in exif:
            return img_bgr
        orientation = exif[orientation_key]
        im = pil_image
        if orientation == 3:
            im = pil_image.rotate(180, expand=True)
        elif orientation == 6:
            im = pil_image.rotate(-90, expand=True)
        elif orientation == 8:
            im = pil_image.rotate(90, expand=True)
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception:
        return img_bgr

def rotate90k(img_bgr: np.ndarray, k: int) -> np.ndarray:
    """k∈{0,1,2,3} → поворот на 0/90/180/270 по часовой."""
    k = int(k) % 4
    if k == 0:
        return img_bgr
    elif k == 1:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    else:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
