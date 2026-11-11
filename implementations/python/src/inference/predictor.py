"""Inferencia con un modelo guardado (docs/architecture.md §8): logit → DOG/NOT_DOG, probabilidad y confianza."""
from pathlib import Path

import numpy as np

from ..core.activations import sigmoid_forward
from ..data.preprocessing import to_float32
from ..model.cnn import CNN

IMAGE_BYTES = 3072          # 3×32×32 uint8, layout plano CHW (dataset.md §6)
CHUNK = 32
LABELS = ("NOT_DOG", "DOG")


class Predictor:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path):
        """Acepta model_final.tb o un checkpoint (se ignoran los `meta.*`)."""
        if not Path(path).is_file():
            raise FileNotFoundError(f"no existe el modelo {path}")
        return cls(CNN.load(path))

    def logits(self, images):
        """images: uint8 (N, 3072) → logits float32 (N,). Normalización idéntica a la del entrenamiento."""
        images = np.asarray(images)
        if images.dtype != np.uint8 or images.ndim != 2 or images.shape[1] != IMAGE_BYTES or images.shape[0] == 0:
            raise ValueError(f"se esperaba uint8 (N>0, {IMAGE_BYTES}), recibió {images.dtype} {images.shape}")
        out = [self.model.forward(to_float32(images[i:i + CHUNK])).reshape(-1).copy()
               for i in range(0, len(images), CHUNK)]
        return np.concatenate(out)

    def predict_batch(self, images):
        """Lista de dicts {prediction, probability, confidence} (3 decimales, architecture.md §8)."""
        z = self.logits(images)
        p = sigmoid_forward(z).astype(np.float64)
        res = []
        for zi, pi in zip(z, p):
            conf = max(pi, 1.0 - pi)
            res.append({"prediction": "DOG" if zi >= 0 else "NOT_DOG",    # decisión por logit, no por σ
                        "probability": round(float(pi), 3), "confidence": round(float(conf), 3)})
        return res

    def predict(self, image):
        return self.predict_batch(np.asarray(image).reshape(1, -1))[0]


def read_images(path, single):
    """Lee un fichero crudo uint8. `single`: exactamente 3072 B; si no, múltiplo no nulo de 3072 B."""
    data = np.fromfile(path, dtype=np.uint8)
    if single and data.size != IMAGE_BYTES:
        raise ValueError(f"{path}: {data.size} bytes, una imagen son {IMAGE_BYTES}")
    if data.size == 0 or data.size % IMAGE_BYTES:
        raise ValueError(f"{path}: {data.size} bytes, no es un múltiplo no nulo de {IMAGE_BYTES}")
    return data.reshape(-1, IMAGE_BYTES)
