import numpy as np
import pytest

from src.layers.dense import Dense
from gradcheck import check_layer


def make(i, o, dtype, seed=0):
    rng = np.random.default_rng(seed)
    d = Dense(i, o, dtype)
    d.set_params(rng.standard_normal((o, i)), rng.standard_normal(o))
    return d, rng


def test_identity_weights_zero_bias():
    d = Dense(4, 4)
    d.set_params(np.eye(4, dtype=np.float32), np.zeros(4, np.float32))
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(d.forward(x), x)


def test_matches_direct_formula():
    d, rng = make(5, 3, np.float64)
    x = rng.standard_normal((4, 5))
    y = d.forward(x)
    for n in range(4):
        for o in range(3):
            assert abs(y[n, o] - (d.bias[o] + sum(d.weight[o, i] * x[n, i] for i in range(5)))) < 1e-12


@pytest.mark.parametrize("shape", [(3, 4, 4), (7, 1152, 64)])
def test_gradcheck(shape):
    n, i, o = (shape[0], shape[1], shape[2]) if len(shape) == 3 else shape
    d, rng = make(i, o, np.float64)
    errs = check_layer(d, rng.standard_normal((n, i)), 1e-6, 1e-6)
    assert set(errs) == {"input", "weight", "bias"}


def test_shape_validation_and_dtype():
    d = Dense(4, 2)
    with pytest.raises(ValueError):
        d.forward(np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError):
        d.backward(np.zeros((3, 2), np.float32))               # sin forward previo
    assert d.forward(np.zeros((3, 4), np.float32)).dtype == np.float32
    d.backward(np.ones((3, 2), np.float32))
    d.zero_grads()
    assert not d.grad_weight.any()
    assert [n for n, _ in d.params()] == ["weight", "bias"] and [n for n, _ in d.grads()] == ["weight", "bias"]


def test_gradcheck_real_dense2_shape():
    d, rng = make(64, 1, np.float64, seed=4)
    check_layer(d, rng.standard_normal((5, 64)), 1e-6, 1e-6)
