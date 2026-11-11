"""Deriva el dataset binario DOG/NOT_DOG, manifiestos y permutaciones a partir de CIFAR-10 binario (docs/dataset.md §5-§10).

Uso: implementations/python/.venv/bin/python -I scripts/prepare_dataset.py
Único lector del archivo original. Lee el tar.gz sin extraerlo a disco. Un solo flujo splitmix64 (D-029).
"""
import csv
import hashlib
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prng import SEED, SplitMix64  # noqa: E402

DATASET = ROOT / "shared" / "dataset"
PERMS = ROOT / "shared" / "permutations"
TAR = DATASET / "raw" / "cifar-10-binary.tar.gz"
BATCH_BYTES = 10000 * 3073
DOG = 5
NEG = [0, 1, 2, 3, 4, 6, 7, 8, 9]
SIZES = {"train": (4000, 4000), "val": (500, 500), "test": (1000, 1000)}      # (perros, no perros)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def quotas(total_neg):
    """Cuota por clase negativa: base + resto repartido en orden ascendente de id (dataset.md §4)."""
    base, rest = divmod(total_neg, len(NEG))
    return {c: base + (1 if i < rest else 0) for i, c in enumerate(NEG)}


def read_pools():
    """→ (imágenes train (50000,3072), clases train, imágenes test, clases test). Valida tamaños y etiquetas."""
    names = [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin"]
    blobs = {}
    with tarfile.open(TAR, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        for n in names + ["batches.meta.txt"]:
            key = f"cifar-10-batches-bin/{n}"
            if key not in members or not members[key].isfile():
                raise SystemExit(f"falta {key} en el archivo")
            blobs[n] = tar.extractfile(members[key]).read()
    meta = blobs["batches.meta.txt"].decode("ascii").split()
    if meta != CLASS_NAMES:
        raise SystemExit(f"batches.meta.txt inesperado: {meta}")
    arrays = []
    for n in names:
        if len(blobs[n]) != BATCH_BYTES:
            raise SystemExit(f"{n}: {len(blobs[n])} B, se esperaban {BATCH_BYTES}")
        rec = np.frombuffer(blobs[n], np.uint8).reshape(10000, 3073)
        if rec[:, 0].max() > 9:
            raise SystemExit(f"{n}: etiqueta fuera de 0-9")
        arrays.append((rec[:, 1:], rec[:, 0]))
    train_x = np.concatenate([a[0] for a in arrays[:5]])
    train_y = np.concatenate([a[1] for a in arrays[:5]])
    return train_x, train_y, arrays[5][0], arrays[5][1]


def select(train_y, test_y):
    """Selección estratificada con un único flujo (dataset.md §5.1). Devuelve índices globales por conjunto, ordenados."""
    rng = SplitMix64(SEED)
    q = {s: quotas(SIZES[s][1]) for s in SIZES}
    for s in SIZES:
        q[s][DOG] = SIZES[s][0]
    sel = {"train": [], "val": [], "test": []}
    for c in range(10):                                       # pool de entrenamiento: clases 0→9
        idx = [int(i) for i in np.flatnonzero(train_y == c)]
        rng.shuffle(idx)
        sel["train"] += idx[: q["train"][c]]
        sel["val"] += idx[q["train"][c]: q["train"][c] + q["val"][c]]
    for c in range(10):                                       # pool de test: clases 0→9
        idx = [int(i) for i in np.flatnonzero(test_y == c)]
        rng.shuffle(idx)
        sel["test"] += idx[: q["test"][c]]
    return {s: sorted(v) for s, v in sel.items()}, q            # orden de fichero: índice global ascendente (D-035)


def source_of(split, g):
    return ("test_batch", g) if split == "test" else (f"data_batch_{g // 10000 + 1}", g % 10000)


def validate(sel, q, train_y, test_y):
    """Comprobaciones de dataset.md §10; aborta si falla alguna."""
    sets = {s: set(v) for s, v in sel.items()}
    assert not (sets["train"] & sets["val"]), "train ∩ val ≠ ∅"
    for s in SIZES:
        assert len(sel[s]) == len(sets[s]) == sum(SIZES[s]), f"{s}: tamaño o duplicados"
    assert max(sets["train"] | sets["val"]) < 50000 and max(sets["test"]) < 10000
    for s, y in (("train", train_y), ("val", train_y), ("test", test_y)):
        cls = np.array([y[g] for g in sel[s]])
        assert (cls == DOG).sum() == SIZES[s][0], f"{s}: nº de perros"
        assert (cls != DOG).sum() == SIZES[s][1], f"{s}: nº de no perros"
        for c in range(10):
            assert (cls == c).sum() == q[s][c], f"{s}: cuota clase {c}"
    assert set(int(i) for i in np.flatnonzero(test_y == DOG)) <= sets["test"], "test debe usar los 1.000 perros del pool de test"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    train_x, train_y, test_x, test_y = read_pools()
    sel, q = select(train_y, test_y)
    validate(sel, q, train_y, test_y)
    (DATASET / "manifests").mkdir(parents=True, exist_ok=True)
    PERMS.mkdir(parents=True, exist_ok=True)
    produced = []
    for split in ("train", "val", "test"):
        x, y = (test_x, test_y) if split == "test" else (train_x, train_y)
        g = np.array(sel[split])
        cls = y[g]
        (DATASET / f"{split}_images.bin").write_bytes(np.ascontiguousarray(x[g]).tobytes())
        (DATASET / f"{split}_labels.bin").write_bytes((cls == DOG).astype(np.uint8).tobytes())
        with open(DATASET / "manifests" / f"{split}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["position", "source", "source_index", "cifar_class", "binary_label"])
            for pos, (gi, c) in enumerate(zip(g, cls)):
                src, si = source_of(split, int(gi))
                w.writerow([pos, src, si, int(c), int(c == DOG)])
        produced += [DATASET / f"{split}_images.bin", DATASET / f"{split}_labels.bin", DATASET / "manifests" / f"{split}.csv"]
    for e in range(1, 21):                                    # permutaciones de época (dataset.md §8)
        perm = list(range(8000))
        SplitMix64(SEED + e).shuffle(perm)
        assert sorted(perm) == list(range(8000))
        p = PERMS / f"train_epoch_{e:03d}.perm"
        p.write_bytes(np.array(perm, "<u4").tobytes())
        produced.append(p)
    ck = DATASET / "CHECKSUMS.sha256"
    keep = [l for l in ck.read_text().splitlines() if l.split("  ", 1)[-1].startswith("raw/")] if ck.exists() else []
    lines = keep + [f"{sha256(p)}  {Path(p).relative_to(DATASET) if DATASET in Path(p).parents else '../permutations/' + Path(p).name}" for p in produced]
    ck.write_text("\n".join(lines) + "\n")
    print({s: len(v) for s, v in sel.items()}, "→", len(produced), "ficheros; checksums en", ck.relative_to(ROOT))


if __name__ == "__main__":
    main()
