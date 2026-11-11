"""Conv2D stride 1, VALID, con bias (docs/model.md §4.1). Estrategia propia: im2col + GEMM (D-001)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel=3, dtype=np.float32, needs_input_grad=True):
        self.c, self.o, self.k, self.dtype = in_channels, out_channels, kernel, np.dtype(dtype)
        self.needs_input_grad = needs_input_grad          # la primera capa puede omitir dX (model.md §4.1)
        self.weight = np.zeros((out_channels, in_channels, kernel, kernel), self.dtype)
        self.bias = np.zeros(out_channels, self.dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._cols = None
        self._x_shape = None

    def set_params(self, weight, bias):
        if weight.shape != self.weight.shape or bias.shape != self.bias.shape:
            raise ValueError("formas de parámetros incorrectas")
        self.weight, self.bias = weight.astype(self.dtype), bias.astype(self.dtype)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.c or x.shape[2] < self.k or x.shape[3] < self.k:
            raise ValueError(f"Conv2D espera (N, {self.c}, H≥{self.k}, W≥{self.k}), recibió {x.shape}")
        n, _, h, w = x.shape
        ho, wo, k = h - self.k + 1, w - self.k + 1, self.k
        win = sliding_window_view(x, (k, k), axis=(2, 3))                    # (N, C, Ho, Wo, K, K)
        self._cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, self.c * k * k)
        self._x_shape = x.shape
        out = self._cols @ self.weight.reshape(self.o, -1).T + self.bias    # (N·Ho·Wo, O)
        return np.ascontiguousarray(out.reshape(n, ho, wo, self.o).transpose(0, 3, 1, 2))

    def backward(self, dy):
        if self._cols is None:
            raise ValueError("backward sin forward previo")
        n, _, h, w = self._x_shape
        ho, wo, k = h - self.k + 1, w - self.k + 1, self.k
        if dy.shape != (n, self.o, ho, wo):
            raise ValueError(f"dY con forma {dy.shape}, se esperaba {(n, self.o, ho, wo)}")
        dym = dy.transpose(0, 2, 3, 1).reshape(n * ho * wo, self.o)
        self.grad_weight = (dym.T @ self._cols).reshape(self.weight.shape)
        self.grad_bias = dym.sum(axis=0)
        if not self.needs_input_grad:
            return None
        dcols = (dym @ self.weight.reshape(self.o, -1)).reshape(n, ho, wo, self.c, k, k)
        dx = np.zeros(self._x_shape, dtype=dy.dtype)
        for u in range(k):                                                   # col2im: acumulación dispersa
            for v in range(k):
                dx[:, :, u:u + ho, v:v + wo] += dcols[:, :, :, :, u, v].transpose(0, 3, 1, 2)
        return dx

    def params(self):
        return [("weight", self.weight), ("bias", self.bias)]

    def grads(self):
        return [("weight", self.grad_weight), ("bias", self.grad_bias)]

    def zero_grads(self):
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
