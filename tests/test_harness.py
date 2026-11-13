"""Utilidades estadísticas del arnés (benchmark/harness/run_benchmark.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "benchmark" / "harness"))
import run_benchmark as rb  # noqa: E402


def test_percentile_matches_numpy():
    v = list(np.random.default_rng(1).normal(size=101))
    for q in (0, 5, 50, 95, 100):
        assert rb.pct(v, q) == pytest.approx(float(np.percentile(v, q)))


def test_outlier_rule_mad_and_10pct():
    assert rb.outliers([10, 10.1, 9.9, 10.05, 10.0]) == [False] * 5
    assert rb.outliers([10, 10.1, 9.9, 10.05, 14.0]) == [False, False, False, False, True]   # fuera de MAD y >10 %
    assert rb.outliers([10, 10, 10, 10, 10.5]) == [False] * 5                              # MAD=0: solo la regla del 10 %
    assert rb.outliers([10, 10, 10, 10, 12]) == [False, False, False, False, True]


def test_describe_basic():
    d = rb.describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d["median"] == 3 and d["mean"] == 3 and d["n"] == 5 and d["std"] == pytest.approx(1.5811388)
