"""ReLU y Sigmoid (docs/model.md §4.2, §4.6). Conservan el dtype de la entrada."""
import numpy as np


def relu_forward(x):
    return np.maximum(x, 0)


def relu_backward(x, dy):
    """dX = dY · 1[x > 0]; derivada en x = 0 definida como 0 (D-008)."""
    return dy * (x > 0)


def sigmoid_forward(z):
    """σ estable: nunca evalúa exp con argumento positivo."""
    z = np.asarray(z)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e)).astype(z.dtype, copy=False)


def sigmoid_backward(out, dy):
    """σ'(z) = σ(1−σ), a partir de la salida ya calculada."""
    return dy * out * (1 - out)
