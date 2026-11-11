"""Diferencias finitas centrales en FP64 (solo pruebas; docs/model.md §10).

g_num = [f(x+h) − f(x−h)] / (2h), h = 1e-5.
error_rel = ‖g_num − g_ana‖₂ / max(‖g_num‖₂, ‖g_ana‖₂, 1e-12)
Para tensores grandes se comprueba una muestra aleatoria de elementos (semilla fija); el error se calcula sobre la muestra.
"""
import numpy as np

H = 1e-5


def rel_error(g_num, g_ana):
    num = np.linalg.norm(g_num - g_ana)
    den = max(np.linalg.norm(g_num), np.linalg.norm(g_ana), 1e-12)
    return float(num / den)


def numeric_grad(f, x, max_elems=400, seed=0):
    """Devuelve (índices planos comprobados, gradiente numérico en ellos). f modifica x in situ vía view."""
    flat = x.reshape(-1)
    idx = np.arange(flat.size)
    if flat.size > max_elems:
        idx = np.sort(np.random.default_rng(seed).choice(flat.size, max_elems, replace=False))
    g = np.empty(len(idx))
    for j, i in enumerate(idx):
        old = flat[i]
        flat[i] = old + H
        fp = f()
        flat[i] = old - H
        fm = f()
        flat[i] = old
        g[j] = (fp - fm) / (2 * H)
    return idx, g


def check_layer(layer, x, tol_input, tol_param, seed=1, max_elems=400):
    """Compara dX y gradientes de parámetros de `layer` con diferencias finitas. Devuelve dict de errores."""
    assert x.dtype == np.float64
    rng = np.random.default_rng(seed)
    out = layer.forward(x)
    r = rng.standard_normal(out.shape)                       # proyección fija: L = Σ out · R
    dx = layer.backward(r)
    errors = {}

    def loss():
        return float(np.sum(layer.forward(x) * r))

    idx, g = numeric_grad(loss, x, max_elems, seed)
    errors["input"] = rel_error(g, dx.reshape(-1)[idx])
    assert errors["input"] < tol_input, f"dX: error relativo {errors['input']:.3e} ≥ {tol_input}"
    analytic = {name: grad.copy() for name, grad in layer.grads()}
    for name, p in layer.params():
        idx, g = numeric_grad(loss, p, max_elems, seed)
        errors[name] = rel_error(g, analytic[name].reshape(-1)[idx])
        assert errors[name] < tol_param, f"d{name}: error relativo {errors[name]:.3e} ≥ {tol_param}"
    return errors
