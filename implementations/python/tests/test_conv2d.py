import numpy as np
import pytest

from src.layers.conv2d import Conv2D
from gradcheck import check_layer


def direct_conv(x, w, b):
    """Referencia literal de model.md §4.1 (bucles directos)."""
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    ho, wo = h - k + 1, wd - k + 1
    y = np.empty((n, o, ho, wo), x.dtype)
    for a in range(n):
        for f in range(o):
            for i in range(ho):
                for j in range(wo):
                    y[a, f, i, j] = b[f] + sum(w[f, ch, u, v] * x[a, ch, i + u, j + v]
                                               for ch in range(c) for u in range(k) for v in range(k))
    return y


def direct_backward(x, w, dy):
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    ho, wo = h - k + 1, wd - k + 1
    dw, db, dx = np.zeros_like(w), np.zeros(o), np.zeros_like(x)
    for a in range(n):
        for f in range(o):
            for i in range(ho):
                for j in range(wo):
                    g = dy[a, f, i, j]
                    db[f] += g
                    for ch in range(c):
                        for u in range(k):
                            for v in range(k):
                                dw[f, ch, u, v] += g * x[a, ch, i + u, j + v]
                                dx[a, ch, i + u, j + v] += g * w[f, ch, u, v]
    return dw, db, dx


def make(c, o, dtype, seed=0):
    rng = np.random.default_rng(seed)
    layer = Conv2D(c, o, 3, dtype)
    layer.set_params(rng.standard_normal((o, c, 3, 3)), rng.standard_normal(o))
    return layer, rng


def test_forward_and_backward_match_direct_loops():
    layer, rng = make(3, 4, np.float64)
    x = rng.standard_normal((2, 3, 7, 6))
    y = layer.forward(x)
    assert np.allclose(y, direct_conv(x, layer.weight, layer.bias), rtol=1e-12, atol=1e-12)
    dy = rng.standard_normal(y.shape)
    dx = layer.backward(dy)
    dw, db, dx_ref = direct_backward(x, layer.weight, dy)
    assert np.allclose(layer.grad_weight, dw, atol=1e-11) and np.allclose(layer.grad_bias, db, atol=1e-11)
    assert np.allclose(dx, dx_ref, atol=1e-11)


def test_zero_params_and_delta_kernel():
    c = Conv2D(1, 1, 3)
    assert not c.forward(np.full((1, 1, 5, 5), 7, np.float32)).any()                   # pesos y bias 0 → 0
    w = np.zeros((1, 1, 3, 3), np.float32)
    w[0, 0, 1, 1] = 1                                                                  # delta central
    c.set_params(w, np.zeros(1, np.float32))
    x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    assert np.array_equal(c.forward(x)[0, 0], x[0, 0, 1:4, 1:4])                       # copia desplazada


@pytest.mark.parametrize("c,o,hw", [(1, 2, 5), (3, 4, 8), (16, 3, 15)])
def test_gradcheck(c, o, hw):
    layer, rng = make(c, o, np.float64)
    errs = check_layer(layer, rng.standard_normal((2, c, hw, hw)), 1e-5, 1e-5)
    assert set(errs) == {"input", "weight", "bias"}


def test_real_shapes_and_fp32():
    for c, o, hw, out in [(3, 16, 32, 30), (16, 32, 15, 13)]:
        layer = Conv2D(c, o, 3)
        y = layer.forward(np.ones((2, c, hw, hw), np.float32))
        assert y.shape == (2, o, out, out) and y.dtype == np.float32
        assert layer.backward(y).shape == (2, c, hw, hw)


def test_input_grad_can_be_skipped_but_param_grads_remain():
    layer = Conv2D(3, 2, 3, needs_input_grad=False)
    y = layer.forward(np.ones((1, 3, 6, 6), np.float32))
    assert layer.backward(np.ones_like(y)) is None and layer.grad_weight.any() and layer.grad_bias.any()


def test_validation():
    layer = Conv2D(3, 2)
    for bad in [np.zeros((1, 2, 8, 8), np.float32), np.zeros((1, 3, 2, 8), np.float32), np.zeros((3, 8, 8), np.float32)]:
        with pytest.raises(ValueError):
            layer.forward(bad)
    layer.forward(np.zeros((1, 3, 8, 8), np.float32))
    with pytest.raises(ValueError):
        layer.backward(np.zeros((1, 2, 5, 5), np.float32))


@pytest.mark.parametrize("c,o,hw", [(3, 16, 32), (16, 32, 15)])
def test_gradcheck_real_shapes(c, o, hw):
    """Formas reales de conv1 y conv2 (muestra aleatoria de 400 elementos por tensor)."""
    layer, rng = make(c, o, np.float64, seed=3)
    check_layer(layer, rng.standard_normal((2, c, hw, hw)), 1e-5, 1e-5)
