"""T1: cada operación del motor (FP32) contra los vectores dorados de shared/test_vectors/ (architecture.md §10, §12)."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.core.activations import relu_backward, relu_forward, sigmoid_backward, sigmoid_forward
from src.core.losses import bce_with_logits
from src.core.tensor_utils import compare, read_tb
from src.layers.conv2d import Conv2D
from src.layers.dense import Dense
from src.layers.flatten import Flatten
from src.layers.maxpool2d import MaxPool2D
from src.model.cnn import CNN
from src.training.optimizer import SGD

VECTORS = Path(__file__).resolve().parents[3] / "shared" / "test_vectors"
INITIAL = VECTORS.parent / "initial_weights" / "initial_weights.tb"
MODEL = (1e-4, 1e-4)                                              # logits, activaciones y pesos tras un paso (§12)
FWD, GRAD, LOSS = (1e-5, 1e-5), (1e-4, 1e-4), (1e-5, 1e-5)     # (atol, rtol), architecture.md §12


def rows():
    with open(VECTORS / "manifest.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(op, t):
    """Ejecuta la operación del motor sobre las entradas del caso. Devuelve {nombre_tensor: (valor, tolerancia)}."""
    x = t.get("input")
    if op == "conv2d":
        w = t["weight"]
        layer = Conv2D(w.shape[1], w.shape[0], 3)
        layer.set_params(w, t["bias"])
        y = layer.forward(x)
        dx = layer.backward(t["grad_output"])
        return {"output": (y, FWD), "grad_input": (dx, GRAD), "grad_weight": (layer.grad_weight, GRAD), "grad_bias": (layer.grad_bias, GRAD)}
    if op == "dense":
        w = t["weight"]
        layer = Dense(w.shape[1], w.shape[0])
        layer.set_params(w, t["bias"])
        y = layer.forward(x)
        dx = layer.backward(t["grad_output"])
        return {"output": (y, FWD), "grad_input": (dx, GRAD), "grad_weight": (layer.grad_weight, GRAD), "grad_bias": (layer.grad_bias, GRAD)}
    if op == "maxpool2d":
        layer = MaxPool2D()
        return {"output": (layer.forward(x), FWD), "grad_input": (layer.backward(t["grad_output"]), GRAD)}
    if op == "flatten":
        layer = Flatten()
        return {"output": (layer.forward(x), FWD), "grad_input": (layer.backward(t["grad_output"]), GRAD)}
    if op == "relu":
        return {"output": (relu_forward(x), FWD), "grad_input": (relu_backward(x, t["grad_output"]), GRAD)}
    if op == "sigmoid":
        s = sigmoid_forward(x)
        return {"output": (s, FWD), "grad_input": (sigmoid_backward(s, t["grad_output"]), GRAD)}
    if op == "bce":
        loss, dz = bce_with_logits(x, t["labels"])
        return {"loss": (np.array([loss]), LOSS), "grad_input": (dz, GRAD)}
    if op == "model":
        model = CNN.load(INITIAL)
        out = {"logits": (model.forward(x), MODEL)}
        out.update({f"a{i}": (model.outputs[i - 1], MODEL) for i in range(1, 10)})
        return out
    if op == "train_step":                                          # T2: forward + BCE + backward + SGD con el lote dorado
        model = CNN()
        model.load_params({k.split(".", 1)[1]: v for k, v in t.items() if k.startswith("weights_init.")})
        loss, dz = bce_with_logits(model.forward(x), t["labels"])
        model.backward(dz)
        grads = {n: g.copy() for n, g in model.grads().items()}
        SGD().step(model.params(), model.grads())
        out = {"loss": (np.array([loss]), LOSS)}
        for n in grads:
            out[f"grads.{n}"] = (grads[n], GRAD)
            out[f"weights_updated.{n}"] = (model.params()[n], MODEL)
        return out
    raise AssertionError(op)


def test_manifest_covers_required_cases():
    ops = {}
    for r in rows():
        ops[r["operation"]] = ops.get(r["operation"], 0) + 1
    assert ops == {"conv2d": 4, "maxpool2d": 4, "relu": 2, "sigmoid": 2, "flatten": 1, "dense": 2, "bce": 3, "model": 1, "train_step": 1}


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["case"])
def test_engine_matches_golden_vector(row):
    path = VECTORS / f"{row['case']}.tb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "vector alterado respecto al manifiesto"
    t = read_tb(path)
    report = []
    for name, (value, (atol, rtol)) in run(row["operation"], t).items():
        ok, worst, pos = compare(value, t[name], atol, rtol)
        report.append(f"{name}: {'ok' if ok else 'FALLO'} exceso máx {worst:.3e} en {pos}")
        assert ok, f"{row['case']} {name}: exceso máximo {worst:.3e} en {pos} (atol={atol}, rtol={rtol})"
    print(row["case"], "; ".join(report))


# --- Criterio adicional, más estricto que el contractual (propuesta P-016) ---------------------------------
# atol = 1e-4 es mayor que muchos gradientes reales (60 % de dense1.weight y 34 % de conv2.weight en train_step
# valen < 1e-4) y que el cambio por paso (lr·g ≤ 1e-2). Medido con mutaciones: un lr erróneo del 10 % lo detecta el
# contrato, pero uno del 1 % o del 0,1 % PASA los vectores con las tolerancias contractuales y FALLA con este criterio.
# Error escalado por tensor = max|a−b| / max|b|; valores medidos del motor ≤ 1,1e-6.
SCALED_TOL = 1e-5


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["case"])
def test_scaled_error_per_tensor(row):
    t = read_tb(VECTORS / f"{row['case']}.tb")
    for name, (value, _) in run(row["operation"], t).items():
        b = t[name].astype(np.float64)
        scale = np.abs(b).max()
        if scale == 0:
            assert not np.asarray(value).any(), f"{row['case']} {name}: esperado todo cero"
            continue
        err = np.abs(np.asarray(value, np.float64) - b).max() / scale
        assert err < SCALED_TOL, f"{row['case']} {name}: error escalado {err:.2e} ≥ {SCALED_TOL}"


def test_t2_weight_update_is_exactly_lr_times_gradient():
    """Δw = w_init − w_updated debe ser lr·g: comprueba directamente el optimizador contra el vector dorado."""
    t = read_tb(VECTORS / "train_step" / "case_001.tb")
    for name in [k.split(".", 1)[1] for k in t if k.startswith("weights_init.")]:
        delta = t[f"weights_init.{name}"].astype(np.float64) - t[f"weights_updated.{name}"].astype(np.float64)
        expected = 0.01 * t[f"grads.{name}"].astype(np.float64)
        scale = np.abs(expected).max()
        assert np.abs(delta - expected).max() < 1e-3 * scale, f"{name}: la actualización del vector no es lr·g"
