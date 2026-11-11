"""Pruebas transversales: PRNG de referencia y regeneración bit a bit de los artefactos compartidos.

Ejecutar con el entorno del motor:  implementations/python/.venv/bin/pytest tests/
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prng import SplitMix64  # noqa: E402


def test_splitmix64_reference_values():
    r = SplitMix64(0)                      # salidas publicadas del algoritmo con estado inicial 0
    assert [r.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_shuffle_is_a_permutation_and_deterministic():
    a, b = list(range(1000)), list(range(1000))
    SplitMix64(20261111).shuffle(a)
    SplitMix64(20261111).shuffle(b)
    assert a == b and sorted(a) == list(range(1000)) and a != list(range(1000))


def test_box_muller_moments():
    r = SplitMix64(7)
    z = [v for _ in range(50000) for v in r.normal_pair()]
    mean = sum(z) / len(z)
    var = sum((v - mean) ** 2 for v in z) / len(z)
    assert abs(mean) < 0.02 and abs(var - 1) < 0.03


def test_artifacts_regenerate_bit_identical(tmp_path):
    """Regenerar en un árbol temporal produce exactamente los mismos ficheros que los versionados."""
    ds = ROOT / "shared" / "dataset" / "train_images.bin"
    if not ds.exists():
        import pytest
        pytest.skip("dataset no preparado (FASE 4): no se puede regenerar train_step/case_002")
    work = tmp_path / "repo"
    (work / "scripts").mkdir(parents=True)
    (work / "implementations").mkdir()
    for f in ("prng.py", "generate_artifacts.py"):
        shutil.copy(ROOT / "scripts" / f, work / "scripts" / f)
    shutil.copytree(ROOT / "implementations" / "python" / "src", work / "implementations" / "python" / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel in ("shared/dataset/train_images.bin", "shared/dataset/train_labels.bin", "shared/permutations/train_epoch_001.perm"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, work / rel)
    subprocess.run([sys.executable, str(work / "scripts" / "generate_artifacts.py")], check=True, capture_output=True)
    for rel in [Path("shared/initial_weights/initial_weights.tb"), Path("shared/test_vectors/manifest.csv")] + \
            sorted(p.relative_to(work) for p in (work / "shared" / "test_vectors").rglob("*.tb")):
        h = lambda base: hashlib.sha256((base / rel).read_bytes()).hexdigest()
        assert h(work) == h(ROOT), f"{rel} difiere tras regenerar"
