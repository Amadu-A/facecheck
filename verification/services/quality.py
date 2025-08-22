# verification/services/quality.py
from __future__ import annotations
import numpy as np

def image_quality_metrics(img) -> dict:
    """Лёгкая диагностика: форма и средняя яркость."""
    if img is None:
        return {"shape": None, "mean": None}
    return {"shape": list(img.shape), "mean": float(np.mean(img))}
