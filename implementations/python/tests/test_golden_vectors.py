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

VECTORS = Path(__file__).resolve().parents[3] / "shared" / "test_vectors"
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
    raise AssertionError(op)


def test_manifest_covers_required_cases():
    ops = {}
    for r in rows():
        ops[r["operation"]] = ops.get(r["operation"], 0) + 1
    assert ops == {"conv2d": 4, "maxpool2d": 4, "relu": 2, "sigmoid": 2, "flatten": 1, "dense": 2, "bce": 3}


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
