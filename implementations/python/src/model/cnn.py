"""CNN-32 Binary Classifier (docs/model.md §3, §5, §6): composición de las 10 capas del contrato."""
import numpy as np

from ..core.initializers import CANONICAL_SHAPES, load_initial_weights
from ..core.tensor_utils import check_shape, read_tb, write_tb
from ..layers.conv2d import Conv2D
from ..layers.dense import Dense
from ..layers.flatten import Flatten
from ..layers.maxpool2d import MaxPool2D
from ..layers.relu import ReLU

INPUT_SHAPE = (3, 32, 32)
# Forma de la salida de cada capa por muestra (docs/model.md §3.1); se comprueba en cada forward.
LAYER_OUT = [(16, 30, 30), (16, 30, 30), (16, 15, 15), (32, 13, 13), (32, 13, 13), (32, 6, 6), (1152,), (64,), (64,), (1,)]


class CNN:
    def __init__(self, dtype=np.float32, input_grad=False):
        """`input_grad=False`: Conv2D_1 no calcula dX (optimización permitida, model.md §4.1).
        Las pruebas de gradiente respecto a la entrada usan `input_grad=True`."""
        self.dtype = np.dtype(dtype)
        self.conv1 = Conv2D(3, 16, 3, dtype, needs_input_grad=input_grad)
        self.conv2 = Conv2D(16, 32, 3, dtype)
        self.dense1 = Dense(1152, 64, dtype)
        self.dense2 = Dense(64, 1, dtype)
        self.layers = [self.conv1, ReLU(), MaxPool2D(), self.conv2, ReLU(), MaxPool2D(), Flatten(),
                       self.dense1, ReLU(), self.dense2]
        self._named = {"conv1": self.conv1, "conv2": self.conv2, "dense1": self.dense1, "dense2": self.dense2}
        self.outputs = []        # salidas de cada capa del último forward (referencias, sin copia): a1…a9 y logits

    # ---- parámetros -------------------------------------------------------
    def params(self):
        """dict nombre canónico → array (el mismo objeto que usa la capa: se actualiza in situ)."""
        return {f"{ln}.{n}": a for ln, layer in self._named.items() for n, a in layer.params()}

    def grads(self):
        return {f"{ln}.{n}": a for ln, layer in self._named.items() for n, a in layer.grads()}

    def zero_grads(self):
        for layer in self._named.values():
            layer.zero_grads()

    def load_params(self, tensors):
        for name, shape in CANONICAL_SHAPES.items():
            if name not in tensors:
                raise ValueError(f"falta el tensor {name}")
            check_shape(name, tensors[name], shape)
        for ln, layer in self._named.items():
            layer.set_params(tensors[f"{ln}.weight"], tensors[f"{ln}.bias"])

    def n_params(self):
        return sum(a.size for a in self.params().values())

    # ---- forward / backward ----------------------------------------------
    def forward(self, x):
        """(N, 3, 32, 32) → logits (N, 1). Sin Sigmoid (model.md §3)."""
        if x.ndim != 4 or x.shape[1:] != INPUT_SHAPE:
            raise ValueError(f"se esperaba (N, 3, 32, 32), recibió {x.shape}")
        if x.dtype != self.dtype:
            raise ValueError(f"se esperaba {self.dtype}, recibió {x.dtype} (sin conversiones silenciosas)")
        self.outputs = []
        for layer, expected in zip(self.layers, LAYER_OUT):
            x = layer.forward(x)
            assert x.shape[1:] == expected, f"{type(layer).__name__}: {x.shape[1:]} ≠ {expected}"
            self.outputs.append(x)
        return x

    def backward(self, dz):
        """Recorre las capas en orden inverso (model.md §6). Devuelve dX de la entrada o None si input_grad=False."""
        if dz.shape != self.outputs[-1].shape:
            raise ValueError(f"dZ con forma {dz.shape}, se esperaba {self.outputs[-1].shape}")
        g = dz
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    # ---- persistencia -----------------------------------------------------
    def save(self, path, meta=None):
        """TensorBundle con los 8 tensores canónicos (+ tensores `meta.*` si se pasan; architecture.md §5.5)."""
        write_tb(path, {**self.params(), **(meta or {})})

    @classmethod
    def load(cls, path, dtype=np.float32, input_grad=False):
        model = cls(dtype, input_grad)
        model.load_params(load_initial_weights(path))
        return model
