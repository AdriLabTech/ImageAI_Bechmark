import numpy as np

from src.core.activations import relu_backward, relu_forward, sigmoid_backward, sigmoid_forward
from gradcheck import H, numeric_grad, rel_error


def test_relu_values_and_zero_convention():
    x = np.array([-2.0, -0.0, 0.0, 3.0], np.float32)
    assert np.array_equal(relu_forward(x), [0, 0, 0, 3])
    assert np.array_equal(relu_backward(x, np.ones(4, np.float32)), [0, 0, 0, 1])   # derivada en 0 = 0


def test_relu_gradcheck():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 1.0, 200) * rng.choice([-1, 1], 200)    # separados del kink (> 2h)
    r = rng.standard_normal(200)
    idx, g = numeric_grad(lambda: float(np.sum(relu_forward(x) * r)), x, 400)
    assert rel_error(g, relu_backward(x, r)[idx]) < 1e-6


def test_sigmoid_properties():
    z = np.array([0.0, 30.0, -30.0, 1000.0, -1000.0, 800.0], np.float32)
    s = sigmoid_forward(z)
    assert np.isfinite(s).all()
    assert s[0] == 0.5 and s[1] > 1 - 1e-6 and 0 <= s[2] < 1e-12 and s[3] == 1.0 and s[4] == 0.0
    zz = np.linspace(-20, 20, 101).astype(np.float32)
    assert np.allclose(sigmoid_forward(-zz), 1 - sigmoid_forward(zz), atol=1e-6)


def test_sigmoid_matches_naive_formula_and_gradcheck():
    z = np.linspace(-15, 15, 301)
    assert np.allclose(sigmoid_forward(z), 1 / (1 + np.exp(-z)), rtol=1e-12, atol=0)
    r = np.random.default_rng(0).standard_normal(301)
    idx, g = numeric_grad(lambda: float(np.sum(sigmoid_forward(z) * r)), z, 400)
    assert rel_error(g, sigmoid_backward(sigmoid_forward(z), r)[idx]) < 1e-6


def test_dtype_preserved():
    assert sigmoid_forward(np.zeros(3, np.float32)).dtype == np.float32
    assert relu_forward(np.zeros(3, np.float64)).dtype == np.float64
