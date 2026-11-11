"""Capa ReLU (envoltorio de core.activations)."""
from ..core.activations import relu_backward, relu_forward


class ReLU:
    def __init__(self):
        self._x = None

    def forward(self, x):
        self._x = x
        return relu_forward(x)

    def backward(self, dy):
        if self._x is None or dy.shape != self._x.shape:
            raise ValueError("backward sin forward previo o dY con forma incorrecta")
        return relu_backward(self._x, dy)

    def params(self):
        return []

    def grads(self):
        return []

    def zero_grads(self):
        pass
