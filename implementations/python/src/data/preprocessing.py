"""Preprocesamiento (docs/dataset.md §9): float32(u8) / 255.0, bit-exacto; sin media/desviación ni augmentation."""
import numpy as np

_255 = np.float32(255.0)


def to_float32(u8):
    """uint8 (N, 3072) o (N, 3, 32, 32) → float32 (N, 3, 32, 32) en [0, 1]. La división IEEE es la misma en los seis lenguajes."""
    if u8.dtype != np.uint8:
        raise ValueError(f"se esperaba uint8, recibió {u8.dtype}")
    return u8.reshape(-1, 3, 32, 32).astype(np.float32) / _255
