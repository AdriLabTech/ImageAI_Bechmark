import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.core.initializers import CANONICAL_SHAPES, load_initial_weights

SHARED = Path(__file__).resolve().parents[3] / "shared" / "initial_weights"
STD = {"conv1.weight": (2 / 27) ** .5, "conv2.weight": (2 / 144) ** .5, "dense1.weight": 1 / 24, "dense2.weight": (2 / 65) ** .5}


def test_checksum_and_shapes():
    p = SHARED / "initial_weights.tb"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == (SHARED / "CHECKSUMS.sha256").read_text().split()[0]
    w = load_initial_weights(p)
    assert sum(v.size for v in w.values()) == 78945
    assert set(w) == set(CANONICAL_SHAPES)


@pytest.mark.parametrize("name,std", STD.items())
def test_he_xavier_statistics(name, std):
    v = load_initial_weights(SHARED / "initial_weights.tb")[name].astype(np.float64)
    tol = 6 / np.sqrt(v.size)                                  # ≈ 6 errores típicos del estimador
    assert abs(v.mean()) < std * tol * 1.5
    assert abs(v.std() / std - 1) < tol


def test_biases_are_zero():
    w = load_initial_weights(SHARED / "initial_weights.tb")
    assert all(not w[k].any() for k in w if k.endswith(".bias"))


def test_rejects_wrong_content(tmp_path):
    from src.core.tensor_utils import write_tb
    w = load_initial_weights(SHARED / "initial_weights.tb")
    bad = dict(w); bad["conv1.weight"] = w["conv1.weight"][:8]
    write_tb(tmp_path / "bad.tb", bad)
    with pytest.raises(ValueError):
        load_initial_weights(tmp_path / "bad.tb")
    bad = dict(w); del bad["dense2.bias"]
    write_tb(tmp_path / "bad2.tb", bad)
    with pytest.raises(ValueError):
        load_initial_weights(tmp_path / "bad2.tb")


def test_rewrite_of_committed_artifacts_is_byte_identical(tmp_path):
    from src.core.tensor_utils import read_tb, write_tb
    vectors = Path(__file__).resolve().parents[3] / "shared" / "test_vectors"
    for src in [SHARED / "initial_weights.tb", *sorted(vectors.rglob("*.tb"))]:
        write_tb(tmp_path / "x.tb", read_tb(src))
        assert (tmp_path / "x.tb").read_bytes() == src.read_bytes(), src.name
