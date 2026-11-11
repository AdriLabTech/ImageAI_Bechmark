"""SGD puro: w ← w − lr·g (docs/training.md §1). Sin momentum, weight decay ni estado."""
import numpy as np

LEARNING_RATE = 0.01


class SGD:
    def __init__(self, lr=LEARNING_RATE, dtype=np.float32):
        self.lr = np.dtype(dtype).type(lr)

    def step(self, params, grads):
        """Actualiza in situ todos los parámetros con los gradientes del lote."""
        if params.keys() != grads.keys():
            raise ValueError("params y grads no tienen los mismos nombres")
        for name, p in params.items():
            if grads[name].shape != p.shape or grads[name].dtype != p.dtype:
                raise ValueError(f"{name}: gradiente con forma/dtype incompatible")
            p -= self.lr * grads[name]
