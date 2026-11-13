"""Evaluador común de calidad (docs/benchmark.md §6.3).

Entrada: `test_logits.csv` de un motor + `test_labels.bin`. Calcula todo en FP64 a partir de los logits FP32 (exactos en el CSV),
con el mismo código para todos los motores. No forma parte del código medido.
Uso: python scripts/evaluate.py --logits run/test_logits.csv --labels shared/dataset/test_labels.bin [--reference ref_logits.csv] [--out q.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_TEST, BINS = 2000, 10


def read_logits(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["position", "logit"]:
        raise ValueError(f"{path}: cabecera {rows[0]}, se esperaba ['position', 'logit']")
    if [int(r[0]) for r in rows[1:]] != list(range(len(rows) - 1)):
        raise ValueError(f"{path}: las posiciones no son 0…N-1 en orden")
    z = np.array([float(r[1]) for r in rows[1:]], dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError(f"{path}: logits no finitos")
    return z


def auc_roc(z, y):
    """AUC por rangos (Mann-Whitney) con rango medio en empates: P(z⁺ > z⁻) + ½·P(z⁺ = z⁻)."""
    order = np.argsort(z, kind="stable")
    zs = z[order]
    ranks = np.empty(len(z))
    i = 0
    while i < len(z):
        j = i
        while j + 1 < len(z) and zs[j + 1] == zs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def quality(z, y):
    e = np.exp(-np.abs(z))
    p = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))   # σ estable en FP64: nunca exp con argumento positivo
    pred = (z >= 0).astype(np.uint8)                   # umbral por logit (training.md §1)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ell = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    idx = np.minimum((p * BINS).astype(int), BINS - 1)
    bins, ece = [], 0.0
    for b in range(BINS):
        m = idx == b
        if m.any():
            mean_p, frac = float(p[m].mean()), float(y[m].mean())
            ece += m.sum() / len(z) * abs(frac - mean_p)
            bins.append([round(mean_p, 9), round(frac, 9), int(m.sum())])
        else:
            bins.append([None, None, 0])
    return {"accuracy": round((tp + tn) / len(z), 9), "precision": round(prec, 9), "recall": round(rec, 9),
            "f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 9), "auc_roc": round(auc_roc(z, y), 9),
            "bce_test": round(float(ell.mean()), 9), "confusion_matrix": [[tn, fp], [fn, tp]],
            "calibration_ece": round(float(ece), 9), "calibration_bins": bins}


def numeric_error(z, ref):
    return {"max_abs_logit_diff": float(np.abs(z - ref).max()), "mean_abs_logit_diff": float(np.abs(z - ref).mean()),
            "flipped_predictions_pct": float(((z >= 0) != (ref >= 0)).mean() * 100)}


def evaluate(logits_csv, labels_bin, reference_csv=None):
    y = np.fromfile(labels_bin, dtype=np.uint8)
    z = read_logits(logits_csv)
    if len(y) != N_TEST or len(z) != N_TEST:
        raise ValueError(f"se esperaban {N_TEST} filas: logits={len(z)}, etiquetas={len(y)}")
    out = quality(z, y)
    if reference_csv:
        out["numeric_error_vs_reference"] = numeric_error(z, read_logits(reference_csv))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = evaluate(a.logits, a.labels, a.reference)
    text = json.dumps(res, indent=2, allow_nan=False) + "\n"
    Path(a.out).write_text(text, encoding="utf-8") if a.out else sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
