"""Carga de pesos iniciales compartidos. Este módulo NO genera aleatoriedad (D-006)."""
from .tensor_utils import check_shape, read_tb

CANONICAL_SHAPES = {
    "conv1.weight": (16, 3, 3, 3), "conv1.bias": (16,),
    "conv2.weight": (32, 16, 3, 3), "conv2.bias": (32,),
    "dense1.weight": (64, 1152), "dense1.bias": (64,),
    "dense2.weight": (1, 64), "dense2.bias": (1,),
}


def load_initial_weights(path):
    """Lee un .tb y valida que tenga exactamente los 8 tensores canónicos, f32 y con su forma."""
    tensors = read_tb(path)
    params = {k: v for k, v in tensors.items() if not k.startswith("meta.")}
    if set(params) != set(CANONICAL_SHAPES):
        raise ValueError(f"tensores distintos de los canónicos: {sorted(set(params) ^ set(CANONICAL_SHAPES))}")
    for name, shape in CANONICAL_SHAPES.items():
        check_shape(name, params[name], shape)
        if params[name].dtype.str != "<f4":
            raise ValueError(f"{name}: dtype {params[name].dtype}, se esperaba float32")
    return params
