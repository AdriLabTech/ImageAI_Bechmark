import numpy as np
import pytest

from src.layers.flatten import Flatten


def test_index_formula_and_roundtrip():
    x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    f = Flatten()
    y = f.forward(x)
    assert y.shape == (2, 60)
    for c, h, w in [(0, 0, 0), (1, 2, 3), (2, 3, 4)]:
        assert y[1, c * 20 + h * 5 + w] == x[1, c, h, w]            # k = c·HW + h·W + w
    assert np.array_equal(f.backward(y), x)


def test_validation():
    with pytest.raises(ValueError):
        Flatten().forward(np.zeros((2, 3)))
    f = Flatten()
    f.forward(np.zeros((2, 3, 4, 5)))
    with pytest.raises(ValueError):
        f.backward(np.zeros((2, 59)))
