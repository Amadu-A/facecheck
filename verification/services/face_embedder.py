# verification/services/face_embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import logging
import numpy as np
import cv2
import onnxruntime as ort

logger = logging.getLogger("app")

def _preprocess_arcface(img_bgr: np.ndarray, size: int = 112) -> np.ndarray:
    """BGR->RGB, resize, CHW, normalize to ArcFace standard."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, 0)        # NCHW
    return img

@dataclass
class FaceEmbedderConfig:
    onnx_path: str
    device: str = "auto"  # "cuda", "cpu"
    providers: Optional[List[str]] = None

class FaceEmbedder:
    """
    Встраиваемая модель эмбеддинга (ArcFace-совместимая ONNX).
    Ожидает вход (1,3,112,112) float32.
    """
    def __init__(self, cfg: FaceEmbedderConfig):
        if cfg.providers:
            providers = cfg.providers
        else:
            if cfg.device == "cuda":
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            elif cfg.device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                # auto
                try:
                    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                    _ = ort.InferenceSession(cfg.onnx_path, providers=providers)
                except Exception:
                    providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(cfg.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"ArcFace ONNX loaded: {cfg.onnx_path} with {self.session.get_providers()}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        x = _preprocess_arcface(face_bgr)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        vec = y[0].astype(np.float32)
        # L2 нормализация
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
        return vec
