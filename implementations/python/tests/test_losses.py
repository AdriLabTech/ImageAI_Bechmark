import math

import numpy as np
import pytest

from src.core.losses import bce_per_sample, bce_with_logits
from gradcheck import numeric_grad, rel_error


def test_known_values():
    for y in (0, 1):
        assert abs(float(bce_per_sample(np.float64(0), y)) - math.log(2)) < 1e-15      # ℓ(0, y) = ln 2
    assert float(bce_per_sample(np.float64(40), 1)) < 1e-15                           # z grande, y=1 → 0
    assert abs(float(bce_per_sample(np.float64(40), 0)) - 40) < 1e-12                 # z grande, y=0 → z


def test_no_overflow_on_extreme_logits():
    z = np.array([[1000], [-1000], [88.0], [-88.0]], np.float32)
    y = np.array([0, 1, 1, 0], np.uint8)
    loss, dz = bce_with_logits(z, y)
    assert np.isfinite(loss) and np.isfinite(dz).all()
    assert abs(float(loss) - (1000 + 1000) / 4) < 1e-3                                # dos errores lineales


def test_equals_naive_formula_in_moderate_range():
    rng = np.random.default_rng(0)
    z = rng.uniform(-8, 8, (32, 1))
    y = rng.integers(0, 2, 32)
    p = 1 / (1 + np.exp(-z[:, 0]))
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert abs(float(bce_with_logits(z, y)[0]) - naive) < 1e-12


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    z = rng.uniform(-5, 5, (32, 1))
    y = rng.integers(0, 2, 32)
    _, dz = bce_with_logits(z, y)
    idx, g = numeric_grad(lambda: float(bce_with_logits(z, y)[0]), z, 400)
    assert rel_error(g, dz.reshape(-1)[idx]) < 1e-6


def test_dtype_and_shape_and_nan_abort():
    z = np.zeros((4, 1), np.float32)
    loss, dz = bce_with_logits(z, np.array([0, 1, 0, 1]))
    assert loss.dtype == np.float32 and dz.dtype == np.float32 and dz.shape == (4, 1)
    assert np.allclose(dz[:, 0], [0.125, -0.125, 0.125, -0.125])
    with pytest.raises(FloatingPointError):
        bce_with_logits(np.array([[np.nan]], np.float32), np.array([1]))
