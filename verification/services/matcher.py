# verification/services/matcher.py
from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger("app")

@dataclass
class MatchResult:
    cosine_sim: float
    verified: bool

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def match(vec_doc: np.ndarray, vec_selfie: np.ndarray, threshold: float = 0.55) -> MatchResult:
    sim = cosine_similarity(vec_doc, vec_selfie)
    ok = sim >= threshold
    logger.info(f"match: cos={sim:.4f} thr={threshold} -> {ok}")
    return MatchResult(cosine_sim=sim, verified=ok)
