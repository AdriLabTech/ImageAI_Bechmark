"""FASE 5: ciclo de entrenamiento, checkpoints, registros y determinismo (training.md §3-§8, §10)."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # mono-hilo (D-002)
    os.environ.setdefault(_v, "1")

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.tensor_utils import read_tb, write_tb
from src.training.checkpoint import CONTRACT_VERSION, load_checkpoint_meta
from src.training.trainer import Trainer

SHARED = Path(__file__).resolve().parents[3] / "shared"
pytestmark = pytest.mark.skipif(not (SHARED / "dataset" / "train_images.bin").exists()
                                or not (SHARED / "permutations" / "train_epoch_001.perm").exists(),
                                reason="dataset/permutaciones no generados (ver README)")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    for name in ("a", "b"):
        Trainer(SHARED, base / name, name, epochs=1, log=lambda *_: None).train()
    return base / "a", base / "b"


def test_determinism_model_final_sha256(two_runs):
    a, b = two_runs
    assert sha(a / "model_final.tb") == sha(b / "model_final.tb")
    assert sha(a / "test_logits.csv") == sha(b / "test_logits.csv")


def test_outputs_exist_and_are_readable(two_runs):
    a, _ = two_runs
    for f in ("checkpoint_epoch_001.tb", "checkpoint_latest.tb", "model_final.tb", "train_log.jsonl",
              "final_metrics.json", "test_logits.csv"):
        assert (a / f).is_file(), f
    assert sha(a / "model_final.tb") == sha(a / "checkpoint_epoch_001.tb") == sha(a / "checkpoint_latest.tb")
    t = read_tb(a / "model_final.tb")
    assert len(t) == 8 + 5 and sum(v.size for k, v in t.items() if not k.startswith("meta.")) == 78945
    meta = load_checkpoint_meta(a / "model_final.tb")
    assert meta["epoch"] == 1 and meta["contract_version"] == CONTRACT_VERSION
    row = json.loads((a / "train_log.jsonl").read_text().splitlines()[0])
    assert set(row) == {"engine", "run", "epoch", "train_loss", "val_loss", "val_accuracy", "epoch_time_s"}
    assert np.float32(row["train_loss"]) == np.float32(meta["train_loss"])
    m = json.loads((a / "final_metrics.json").read_text())
    assert {"val_loss", "val_accuracy", "test_loss", "test_accuracy"} <= set(m)


def test_test_logits_csv_format(two_runs):
    lines = (two_runs[0] / "test_logits.csv").read_text().splitlines()
    assert lines[0] == "position,logit" and len(lines) == 2001
    for i, line in enumerate(lines[1:]):
        pos, z = line.split(",")
        assert int(pos) == i
        assert "%.9g" % np.float32(z) == z                              # ida y vuelta exacta de un FP32


def test_one_epoch_reduces_loss_and_is_finite(two_runs):
    row = json.loads((two_runs[0] / "train_log.jsonl").read_text().splitlines()[0])
    assert all(np.isfinite(row[k]) for k in ("train_loss", "val_loss", "val_accuracy"))
    assert row["train_loss"] < 0.6931   # por debajo de ln 2 tras una época (convergencia mínima)


def test_initial_weights_loaded_bit_identical(tmp_path):
    t = Trainer(SHARED, tmp_path, epochs=1)
    model = t._load_model()
    ref = read_tb(t.initial_weights)
    assert all(model.params()[k].tobytes() == ref[k].tobytes() for k in model.params())


def test_nan_aborts_training(tmp_path):
    w = read_tb(SHARED / "initial_weights" / "initial_weights.tb")
    w["dense2.bias"] = np.array([np.nan], dtype="<f4")
    bad = tmp_path / "nan.tb"
    write_tb(bad, w)
    with pytest.raises(FloatingPointError):
        Trainer(SHARED, tmp_path / "out", epochs=1, initial_weights=bad, log=lambda *_: None).train()
