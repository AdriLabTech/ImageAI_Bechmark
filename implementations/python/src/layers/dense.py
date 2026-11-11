"""Capa densa Y = X·Wᵀ + B, W en forma (O, I) (docs/model.md §4.5)."""
import numpy as np


class Dense:
    def __init__(self, in_features, out_features, dtype=np.float32):
        self.i, self.o, self.dtype = in_features, out_features, np.dtype(dtype)
        self.weight = np.zeros((out_features, in_features), self.dtype)
        self.bias = np.zeros(out_features, self.dtype)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._x = None

    def set_params(self, weight, bias):
        if weight.shape != self.weight.shape or bias.shape != self.bias.shape:
            raise ValueError("formas de parámetros incorrectas")
        self.weight, self.bias = weight.astype(self.dtype), bias.astype(self.dtype)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.i:
            raise ValueError(f"Dense espera (N, {self.i}), recibió {x.shape}")
        self._x = x
        return x @ self.weight.T + self.bias

    def backward(self, dy):
        if self._x is None or dy.shape != (self._x.shape[0], self.o):
            raise ValueError("backward sin forward previo o dY con forma incorrecta")
        self.grad_weight = dy.T @ self._x
        self.grad_bias = dy.sum(axis=0)
        return dy @ self.weight

    def params(self):
        return [("weight", self.weight), ("bias", self.bias)]

    def grads(self):
        return [("weight", self.grad_weight), ("bias", self.grad_bias)]

    def zero_grads(self):
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
