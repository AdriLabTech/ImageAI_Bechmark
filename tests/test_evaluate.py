"""Evaluador común (scripts/evaluate.py): valores calculados a mano y contra una implementación independiente."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev  # noqa: E402


def test_hand_computed_small_case():
    z = np.array([2.0, 1.0, -0.5, 0.0, -2.0, 3.0])
    y = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    q = ev.quality(z, y)
    # pred = [1,1,0,1,0,1]: TP=2 (0,5)... y=1 en 0,2,5 → TP 2 (0,5), FN 1 (2); FP 2 (1,3); TN 1 (4)
    assert q["confusion_matrix"] == [[1, 2], [1, 2]]
    assert q["accuracy"] == pytest.approx(3 / 6) and q["precision"] == pytest.approx(0.5)
    assert q["recall"] == pytest.approx(2 / 3) and q["f1"] == pytest.approx(4 / 7)
    # AUC: pares (pos, neg): pos {2,-0.5,3} vs neg {1,0,-2}: 2>1,2>0,2>-2 ✓✓✓; -0.5>-2 ✓ only; 3 ✓✓✓ → 7/9
    assert q["auc_roc"] == pytest.approx(7 / 9)


def test_auc_ties_and_extremes():
    assert ev.auc_roc(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5          # empate → ½
    assert ev.auc_roc(np.array([0.0, 1.0]), np.array([0, 1])) == 1.0
    assert ev.auc_roc(np.array([1.0, 0.0]), np.array([0, 1])) == 0.0


def test_auc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    z = np.round(rng.normal(size=200), 1)                                      # muchos empates
    y = (rng.random(200) < 0.5).astype(np.uint8)
    pos, neg = z[y == 1], z[y == 0]
    brute = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))
    assert ev.auc_roc(z, y) == pytest.approx(brute, abs=1e-12)


def test_calibration_perfect_and_extreme_logits():
    z = np.array([-1e4, 1e4, -1e4, 1e4])                                       # saturación sin desbordar
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    q = ev.quality(z, y)
    assert q["calibration_ece"] == pytest.approx(0, abs=1e-9) and q["bce_test"] == pytest.approx(0, abs=1e-9)
    assert sum(b[2] for b in q["calibration_bins"]) == 4


def test_numeric_error():
    e = ev.numeric_error(np.array([1.0, -1.0, 0.5]), np.array([1.5, 1.0, 0.5]))
    assert e["max_abs_logit_diff"] == 2.0 and e["flipped_predictions_pct"] == pytest.approx(100 / 3)


def test_read_logits_rejects_bad_files(tmp_path):
    for body in ("position,logit\n1,0.5\n", "pos,logit\n0,0.5\n", "position,logit\n0,nan\n"):
        f = tmp_path / "x.csv"
        f.write_text(body)
        with pytest.raises(ValueError):
            ev.read_logits(f)
