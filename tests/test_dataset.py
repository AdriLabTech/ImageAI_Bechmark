"""Verificación independiente del dataset derivado (docs/dataset.md §4-§10). Se omite si aún no se ha preparado.

Relee el tar.gz original y las salidas de scripts/prepare_dataset.py y comprueba cuotas, solapamientos, píxeles,
permutaciones y checksums. Ejecutar:  implementations/python/.venv/bin/pytest tests/test_dataset.py
"""
import csv
import hashlib
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DS, PERMS = ROOT / "shared" / "dataset", ROOT / "shared" / "permutations"
TAR = DS / "raw" / "cifar-10-binary.tar.gz"
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = pytest.mark.skipif(not (DS / "train_images.bin").exists() or not TAR.exists(), reason="dataset no preparado")

DOG = 5
QUOTA = {  # dataset.md §4 (negativas por clase) + positivas
    "train": {0: 445, 1: 445, 2: 445, 3: 445, 4: 444, 5: 4000, 6: 444, 7: 444, 8: 444, 9: 444},
    "val": {0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 500, 6: 55, 7: 55, 8: 55, 9: 55},
    "test": {0: 112, 1: 111, 2: 111, 3: 111, 4: 111, 5: 1000, 6: 111, 7: 111, 8: 111, 9: 111},
}
SIZE = {"train": 8000, "val": 1000, "test": 2000}


def manifest(split):
    with open(DS / "manifests" / f"{split}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def raw():
    """Registros originales indexados por (source, source_index) → (clase, 3072 bytes)."""
    out = {}
    with tarfile.open(TAR, "r:gz") as tar:
        for n in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            rec = np.frombuffer(tar.extractfile(f"cifar-10-batches-bin/{n}.bin").read(), np.uint8).reshape(10000, 3073)
            out[n] = rec
    return out


@pytest.mark.parametrize("split", SIZE)
def test_files_sizes_and_quotas(split):
    rows = manifest(split)
    assert len(rows) == SIZE[split]
    assert (DS / f"{split}_images.bin").stat().st_size == SIZE[split] * 3072
    assert (DS / f"{split}_labels.bin").stat().st_size == SIZE[split]
    assert dict(Counter(int(r["cifar_class"]) for r in rows)) == QUOTA[split]
    assert [int(r["position"]) for r in rows] == list(range(SIZE[split]))
    labels = np.fromfile(DS / f"{split}_labels.bin", np.uint8)
    assert labels.tolist() == [int(r["binary_label"]) for r in rows]
    assert all(int(r["binary_label"]) == (int(r["cifar_class"]) == DOG) for r in rows)
    assert int(labels.sum()) * 2 == SIZE[split]                                   # balance 50/50 exacto


def test_no_overlaps_and_correct_pools():
    keys = {s: [(r["source"], int(r["source_index"])) for r in manifest(s)] for s in SIZE}
    for s in SIZE:
        assert len(set(keys[s])) == len(keys[s]), f"{s}: duplicados"
    assert not set(keys["train"]) & set(keys["val"])
    assert all(src == "test_batch" for src, _ in keys["test"])
    assert all(src.startswith("data_batch_") for src, _ in keys["train"] + keys["val"])
    assert not set(keys["test"]) & (set(keys["train"]) | set(keys["val"]))
    assert len({k for k in keys["test"]}) == 2000 and sum(1 for r in manifest("test") if r["cifar_class"] == "5") == 1000


@pytest.mark.parametrize("split", SIZE)
def test_order_is_ascending_source_index(split):
    k = [(r["source"], int(r["source_index"])) for r in manifest(split)]
    assert k == sorted(k)                                                          # D-035


@pytest.mark.parametrize("split", SIZE)
def test_every_image_equals_original_record(split, raw):
    """Los 3072 bytes y la clase coinciden con el registro original de CIFAR-10 (todas las filas)."""
    imgs = np.fromfile(DS / f"{split}_images.bin", np.uint8).reshape(-1, 3072)
    for pos, r in enumerate(manifest(split)):
        rec = raw[r["source"]][int(r["source_index"])]
        assert rec[0] == int(r["cifar_class"])
        assert np.array_equal(imgs[pos], rec[1:]), f"{split}[{pos}] difiere del original"


def test_permutations_are_exact_distinct_and_regenerable():
    from prng import SEED, SplitMix64
    seen = set()
    for e in range(1, 21):
        p = np.fromfile(PERMS / f"train_epoch_{e:03d}.perm", "<u4")
        assert p.size == 8000 and int(p.sum()) == 8000 * 7999 // 2 and len(set(p.tolist())) == 8000
        ref = list(range(8000)); SplitMix64(SEED + e).shuffle(ref)
        assert p.tolist() == ref
        seen.add(hashlib.sha256(p.tobytes()).hexdigest())
    assert len(seen) == 20                                                         # todas distintas
    labels = np.fromfile(DS / "train_labels.bin", np.uint8)
    for e in (1, 7, 20):                                                           # el barajado conserva la composición
        assert int(labels[np.fromfile(PERMS / f"train_epoch_{e:03d}.perm", "<u4")].sum()) == 4000


def test_checksums_file_matches_disk():
    lines = (DS / "CHECKSUMS.sha256").read_text().splitlines()
    assert any(l.endswith("raw/cifar-10-binary.tar.gz") for l in lines)
    assert len(lines) == 1 + 3 * 3 + 20
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((DS / rel).read_bytes()).hexdigest() == digest, rel
