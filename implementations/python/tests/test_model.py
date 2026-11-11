import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.core.initializers import CANONICAL_SHAPES
from src.core.losses import bce_with_logits
from src.core.tensor_utils import assert_close, read_tb
from src.model.cnn import CNN, LAYER_OUT
from src.training.optimizer import SGD

INITIAL = Path(__file__).resolve().parents[3] / "shared" / "initial_weights" / "initial_weights.tb"


def images(n, seed=0):
    u8 = np.random.default_rng(seed).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)
    return u8.astype(np.float32) / np.float32(255)


def labels(n, seed=1):
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def test_parameter_count_and_canonical_names():
    m = CNN.load(INITIAL)
    assert m.n_params() == 78945
    assert {k: v.shape for k, v in m.params().items()} == CANONICAL_SHAPES
    assert {k: v.shape for k, v in m.grads().items()} == CANONICAL_SHAPES


def test_layer_by_layer_shapes_match_contract_table():
    m = CNN.load(INITIAL)
    z = m.forward(images(5))
    assert z.shape == (5, 1) and z.dtype == np.float32
    assert [o.shape[1:] for o in m.outputs] == LAYER_OUT
    assert [o.shape for o in m.outputs][:3] == [(5, 16, 30, 30), (5, 16, 30, 30), (5, 16, 15, 15)]
    assert m.outputs[5].shape == (5, 32, 6, 6) and m.outputs[6].shape == (5, 1152)
    assert all(o.dtype == np.float32 for o in m.outputs)


def test_forward_is_batch_independent():
    m = CNN.load(INITIAL)
    x = images(6)
    z = m.forward(x).copy()
    one = np.concatenate([m.forward(x[i:i + 1]) for i in range(6)])
    assert_close(z, one, 1e-5, 1e-5, "forward por lotes vs por muestras")


def test_batch_gradient_is_mean_of_per_sample_gradients():
    """Coherencia model.md §6 + §8.2: dZ = (σ−y)/N → los gradientes del lote son la media de los individuales."""
    m = CNN.load(INITIAL)
    x, y = images(8), labels(8)
    loss, dz = bce_with_logits(m.forward(x), y)
    m.backward(dz)
    batch = {k: v.copy() for k, v in m.grads().items()}
    acc = {k: np.zeros_like(v) for k, v in batch.items()}
    losses = []
    for i in range(8):
        l, d = bce_with_logits(m.forward(x[i:i + 1]), y[i:i + 1])
        m.backward(d)
        losses.append(l)
        for k, g in m.grads().items():
            acc[k] += g / 8
    assert abs(float(loss) - float(np.mean(losses))) < 1e-5
    for k in batch:
        assert_close(batch[k], acc[k], 1e-5, 1e-4, k)


def test_validation_of_input_and_dz():
    m = CNN.load(INITIAL)
    for bad in [np.zeros((2, 3, 31, 32), np.float32), np.zeros((2, 1, 32, 32), np.float32), np.zeros((3, 32, 32), np.float32)]:
        with pytest.raises(ValueError):
            m.forward(bad)
    with pytest.raises(ValueError):
        m.forward(np.zeros((2, 3, 32, 32), np.float64))             # sin conversiones silenciosas
    m.forward(images(2))
    with pytest.raises(ValueError):
        m.backward(np.zeros((3, 1), np.float32))


def test_everything_stays_float32_through_a_training_step():
    m = CNN.load(INITIAL)
    loss, dz = bce_with_logits(m.forward(images(4)), labels(4))
    m.backward(dz)
    SGD().step(m.params(), m.grads())
    assert loss.dtype == np.float32 and dz.dtype == np.float32
    assert all(g.dtype == np.float32 for g in m.grads().values()) and all(p.dtype == np.float32 for p in m.params().values())


def test_input_grad_flag_only_skips_dx():
    x, y = images(3), labels(3)
    a, b = CNN.load(INITIAL), CNN.load(INITIAL, input_grad=True)
    for m in (a, b):
        _, dz = bce_with_logits(m.forward(x), y)
        out = m.backward(dz)
        m.dx = out
    assert a.dx is None and b.dx.shape == x.shape
    for k in a.grads():
        assert np.array_equal(a.grads()[k], b.grads()[k])


def test_save_load_roundtrip_and_determinism(tmp_path):
    m = CNN.load(INITIAL)
    m.save(tmp_path / "a.tb")
    assert (tmp_path / "a.tb").read_bytes() == INITIAL.read_bytes()   # ida y vuelta byte-idéntica
    meta = {"meta.epoch": np.array([3], np.uint32)}
    m.save(tmp_path / "b.tb", meta)
    assert read_tb(tmp_path / "b.tb")["meta.epoch"][0] == 3 and CNN.load(tmp_path / "b.tb").n_params() == 78945
    outs = []
    for _ in range(2):                                                  # misma entrada → bits idénticos
        outs.append(hashlib.sha256(CNN.load(INITIAL).forward(images(4)).tobytes()).hexdigest())
    assert outs[0] == outs[1]


def test_load_params_validates():
    m = CNN()
    w = read_tb(INITIAL)
    del w["conv2.bias"]
    with pytest.raises(ValueError):
        m.load_params(w)
    w = read_tb(INITIAL)
    w["dense1.weight"] = w["dense1.weight"][:10]
    with pytest.raises(ValueError):
        m.load_params(w)


def test_sgd_descends_on_a_fixed_batch():
    """Sanidad de que backward + SGD(lr 0,01) reducen la pérdida: memorizar un lote pequeño. No es una métrica de calidad."""
    m = CNN.load(INITIAL)
    opt = SGD()
    x, y = images(16, seed=5), labels(16, seed=6)
    first = last = None
    for step in range(60):
        loss, dz = bce_with_logits(m.forward(x), y)
        m.backward(dz)
        opt.step(m.params(), m.grads())
        first = float(loss) if first is None else first
        last = float(loss)
    print(f"pérdida {first:.4f} → {last:.4f} en 60 pasos sobre 16 imágenes")
    assert last < 0.8 * first


def test_sgd_rejects_mismatched_inputs():
    m = CNN.load(INITIAL)
    g = m.grads()
    g.pop("conv1.bias")
    with pytest.raises(ValueError):
        SGD().step(m.params(), g)
