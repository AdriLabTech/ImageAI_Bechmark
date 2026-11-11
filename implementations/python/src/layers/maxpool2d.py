"""MaxPool2D 2×2, stride 2, floor (docs/model.md §4.3)."""
import numpy as np


class MaxPool2D:
    def __init__(self):
        self._argmax = None
        self._in_shape = None

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] < 2 or x.shape[3] < 2:
            raise ValueError(f"MaxPool2D espera (N, C, H≥2, W≥2), recibió {x.shape}")
        n, c, h, w = x.shape
        ho, wo = h // 2, w // 2
        # Recorta la fila/columna sobrante (dimensión impar) y deja la ventana como último eje en orden (r, s).
        win = x[:, :, : 2 * ho, : 2 * wo].reshape(n, c, ho, 2, wo, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, ho, wo, 4)
        self._argmax = np.argmax(win, axis=-1)           # primer máximo en orden raster (D-008)
        self._in_shape = x.shape
        return np.take_along_axis(win, self._argmax[..., None], axis=-1)[..., 0]

    def backward(self, dy):
        if self._argmax is None or dy.shape != self._argmax.shape:
            raise ValueError("backward sin forward previo o dY con forma incorrecta")
        n, c, h, w = self._in_shape
        ho, wo = h // 2, w // 2
        win = np.zeros((n, c, ho, wo, 4), dtype=dy.dtype)
        np.put_along_axis(win, self._argmax[..., None], dy[..., None], axis=-1)
        dx = np.zeros(self._in_shape, dtype=dy.dtype)    # fila/columna descartada = 0
        dx[:, :, : 2 * ho, : 2 * wo] = win.reshape(n, c, ho, wo, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, 2 * ho, 2 * wo)
        return dx

    def params(self):
        return []

    def grads(self):
        return []

    def zero_grads(self):
        pass
