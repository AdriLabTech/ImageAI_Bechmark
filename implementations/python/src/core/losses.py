"""Binary Cross-Entropy con logits, estable (docs/model.md §8)."""
import numpy as np

from .activations import sigmoid_forward


def bce_per_sample(z, y):
    """ℓ(z, y) = max(z, 0) − z·y + log(1 + e^(−|z|)). No evalúa nunca log σ."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def bce_with_logits(z, y):
    """Devuelve (pérdida media escalar, dZ = (σ(z) − y)/N) con la forma de z.

    La suma sobre el lote se hace en orden de n creciente y en el dtype de z (cumsum es secuencial).
    """
    z = np.asarray(z)
    n = z.shape[0]
    y = np.asarray(y).reshape(z.shape).astype(z.dtype)
    if n == 0:
        raise ValueError("lote vacío")
    ell = bce_per_sample(z, y).reshape(n)
    loss = np.cumsum(ell, dtype=z.dtype)[-1] / z.dtype.type(n)
    dz = (sigmoid_forward(z) - y) / z.dtype.type(n)
    if not np.isfinite(loss):
        raise FloatingPointError("la pérdida no es finita (NaN/Inf): se aborta (model.md §9)")
    return loss, dz.astype(z.dtype, copy=False)
