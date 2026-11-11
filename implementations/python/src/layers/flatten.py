"""Flatten NCHW → (N, C·H·W) (docs/model.md §4.4). Solo reshape, sin aritmética."""


class Flatten:
    def __init__(self):
        self._shape = None

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Flatten espera (N, C, H, W), recibió {x.shape}")
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        if self._shape is None or dy.shape != (self._shape[0], int(self._shape[1] * self._shape[2] * self._shape[3])):
            raise ValueError("backward sin forward previo o dY con forma incorrecta")
        return dy.reshape(self._shape)

    def params(self):
        return []

    def grads(self):
        return []

    def zero_grads(self):
        pass
