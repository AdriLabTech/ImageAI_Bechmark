import numpy as np
import pytest

from src.layers.maxpool2d import MaxPool2D
from gradcheck import check_layer


def direct_pool(x):
    """Referencia literal de model.md §4.3: bucles directos, primer máximo en orden raster."""
    n, c, h, w = x.shape
    ho, wo = h // 2, w // 2
    y = np.empty((n, c, ho, wo), x.dtype)
    arg = np.empty((n, c, ho, wo, 2), int)
    for a in range(n):
        for b in range(c):
            for i in range(ho):
                for j in range(wo):
                    best, pos = None, None
                    for r in range(2):
                        for s in range(2):
                            v = x[a, b, 2 * i + r, 2 * j + s]
                            if best is None or v > best:        # estricto: conserva el primero en empates
                                best, pos = v, (r, s)
                    y[a, b, i, j], arg[a, b, i, j] = best, pos
    return y, arg


def distinct(shape, seed=0):
    """Valores distintos y separados 0,01 ≫ 2h: sin empates ni cruces de kink en el gradcheck."""
    n = int(np.prod(shape))
    return (np.random.default_rng(seed).permutation(n) * 0.01).reshape(shape) - n * 0.005


@pytest.mark.parametrize("shape", [(2, 3, 30, 30), (2, 4, 13, 13), (1, 2, 6, 7)])
def test_matches_direct_reference_and_shapes(shape):
    x = distinct(shape).astype(np.float32)
    p = MaxPool2D()
    y = p.forward(x)
    ref, _ = direct_pool(x)
    assert y.shape == (shape[0], shape[1], shape[2] // 2, shape[3] // 2)
    assert np.array_equal(y, ref)


def test_odd_dimension_discards_last_row_and_column():
    x = np.zeros((1, 1, 13, 13), np.float32)
    x[0, 0, 12, :] = 100.0                                        # fila 12: debe ignorarse
    x[0, 0, :, 12] = 100.0                                        # columna 12: ídem
    p = MaxPool2D()
    assert p.forward(x).max() == 0
    dx = p.backward(np.ones((1, 1, 6, 6), np.float32))
    assert dx.shape == (1, 1, 13, 13) and not dx[0, 0, 12, :].any() and not dx[0, 0, :, 12].any()


def test_ties_go_to_first_in_raster_order():
    x = np.zeros((1, 1, 4, 4), np.float32)
    x[0, 0, 0:2, 0:2] = [[5, 5], [5, 5]]                          # empate total → (0, 0)
    x[0, 0, 0:2, 2:4] = [[1, 7], [7, 2]]                          # empate (0,1) y (1,0) → (0, 1)
    x[0, 0, 2:4, 0:2] = [[3, 3], [9, 9]]                          # empate (1,0) y (1,1) → (1, 0)
    x[0, 0, 2:4, 2:4] = [[0, 0], [0, 0]]                          # todo igual a 0 → (0, 0)
    p = MaxPool2D()
    p.forward(x)
    dx = p.backward(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))
    expected = np.zeros((4, 4), np.float32)
    expected[0, 0] = 1        # bloque (0,0): empate total → posición (r,s)=(0,0)
    expected[0, 3] = 2        # bloque (0,1): máximos en (0,1) y (1,0) → gana (0,1) = fila 0, columna 3
    expected[3 - 1 + 1, 0] = 3  # bloque (1,0): máximos en (1,0) y (1,1) → gana (1,0) = fila 3, columna 0
    expected[2, 2] = 4        # bloque (1,1): todo igual → (0,0) = fila 2, columna 2
    assert np.array_equal(dx[0, 0], expected)
    assert np.array_equal(p.forward(x), direct_pool(x)[0])   # y la referencia directa coincide también con empates


def test_gradient_preserves_total_mass():
    x = distinct((2, 3, 13, 13)).astype(np.float32)
    p = MaxPool2D()
    p.forward(x)
    dy = np.random.default_rng(0).standard_normal((2, 3, 6, 6)).astype(np.float32)
    assert abs(float(p.backward(dy).sum()) - float(dy.sum())) < 1e-4


@pytest.mark.parametrize("shape", [(2, 3, 30, 30), (2, 3, 13, 13)])
def test_gradcheck(shape):
    check_layer(MaxPool2D(), distinct(shape), 1e-5, 1e-5)


def test_validation():
    with pytest.raises(ValueError):
        MaxPool2D().forward(np.zeros((1, 1, 1, 8), np.float32))
    p = MaxPool2D()
    p.forward(np.zeros((1, 1, 4, 4), np.float32))
    with pytest.raises(ValueError):
        p.backward(np.zeros((1, 1, 3, 3), np.float32))
