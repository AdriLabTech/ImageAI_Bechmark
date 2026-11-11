"""Gradcheck FP64 del MODELO COMPLETO (+BCE) respecto a los 8 bloques de parámetros y a la entrada (model.md §10)."""
from pathlib import Path

import numpy as np
import pytest

from src.core.losses import bce_with_logits
from src.core.tensor_utils import read_tb
from src.model.cnn import CNN
from gradcheck import numeric_grad, rel_error

INITIAL = Path(__file__).resolve().parents[3] / "shared" / "initial_weights" / "initial_weights.tb"
TOL = {"conv1.weight": 1e-5, "conv1.bias": 1e-5, "conv2.weight": 1e-5, "conv2.bias": 1e-5,
       "dense1.weight": 1e-6, "dense1.bias": 1e-6, "dense2.weight": 1e-6, "dense2.bias": 1e-6}
TOL_INPUT = 1e-5


def build(seed, n=3):
    rng = np.random.default_rng(seed)
    m = CNN(np.float64, input_grad=True)
    m.load_params({k: v.astype(np.float64) for k, v in read_tb(INITIAL).items()})
    x = rng.integers(0, 256, (n, 3, 32, 32)).astype(np.float64) / 255
    y = rng.integers(0, 2, n)
    return m, x, y


H_RECHECK = 1e-7     # paso menor para descartar cruces de kink (ReLU/MaxPool); ruido FP64 ≈ 1e-16/1e-7 = 1e-9


def run_check(seed):
    """Devuelve (errores finales, tensores que necesitaron repetir con h menor)."""
    m, x, y = build(seed)
    loss_fn = lambda: float(bce_with_logits(m.forward(x), y)[0])
    _, dz = bce_with_logits(m.forward(x), y)
    dx = m.backward(dz)
    grads = {k: v.copy() for k, v in m.grads().items()}
    targets = {**m.params(), "input": x}
    analytic = {**grads, "input": dx}
    tol = {**TOL, "input": TOL_INPUT}
    errors, rechecked = {}, []
    for name, t in targets.items():
        idx, g = numeric_grad(loss_fn, t, max_elems=150, seed=seed)
        errors[name] = rel_error(g, analytic[name].reshape(-1)[idx])
        if errors[name] >= tol[name]:                      # posible cruce de kink: repetir los mismos elementos con h menor
            _, g2 = numeric_grad(loss_fn, t, max_elems=150, seed=seed, h=H_RECHECK)
            errors[name] = rel_error(g2, analytic[name].reshape(-1)[idx])
            rechecked.append(name)
    return errors, rechecked


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_full_network_gradcheck(seed):
    errors, rechecked = run_check(seed)
    print({k: f"{v:.2e}" for k, v in errors.items()}, "| repetidos con h=1e-7:", rechecked)
    for name, tol in {**TOL, "input": TOL_INPUT}.items():
        assert errors[name] < tol, f"{name}: error relativo {errors[name]:.3e} ≥ {tol} (también con h={H_RECHECK})"
    assert len(rechecked) <= 2, "demasiados tensores con cruces de kink: sospechar de un error real"
