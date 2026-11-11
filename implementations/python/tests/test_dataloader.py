"""Pruebas del cargador de datos. Las que necesitan el dataset real se omiten si aún no se ha preparado."""
from pathlib import Path

import numpy as np
import pytest

from src.data.dataloader import EPOCHS, Dataset, load_permutation
from src.data.preprocessing import to_float32

SHARED = Path(__file__).resolve().parents[3] / "shared"
needs_data = pytest.mark.skipif(not (SHARED / "dataset" / "train_images.bin").exists(), reason="dataset no preparado")


def test_normalization_is_exact_division():
    u8 = np.arange(256, dtype=np.uint8)
    x = to_float32(np.tile(u8, 12).reshape(1, 3072))
    assert x.dtype == np.float32 and x.shape == (1, 3, 32, 32)
    # float32(u8)/255.0 es la división IEEE correctamente redondeada: debe igualar a redondear la división exacta en f64
    assert np.array_equal(x.reshape(-1)[:256], (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32))
    assert x.min() == 0.0 and x.max() == 1.0


def test_normalization_rejects_non_uint8():
    with pytest.raises(ValueError):
        to_float32(np.zeros((1, 3072), np.float32))


def make(tmp_path, n=8000, split="train"):
    d = tmp_path / "dataset"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 256, n * 3072, dtype=np.uint8).tofile(d / f"{split}_images.bin")
    rng.integers(0, 2, n, dtype=np.uint8).tofile(d / f"{split}_labels.bin")
    return tmp_path


def test_dataset_validates_sizes_and_labels(tmp_path):
    ds = Dataset(make(tmp_path), "train")
    assert len(ds) == 8000 and ds.images.shape == (8000, 3072)
    x, y = ds.batch([0, 7999, 5])
    assert x.shape == (3, 3, 32, 32) and x.dtype == np.float32 and y.shape == (3,)
    with pytest.raises(IndexError):
        ds.batch([8000])
    with pytest.raises(IndexError):
        ds.batch([-1])
    np.fromfile(tmp_path / "dataset" / "train_labels.bin", np.uint8)[:7999].tofile(tmp_path / "dataset" / "train_labels.bin")
    with pytest.raises(ValueError):
        Dataset(tmp_path, "train")


def test_label_out_of_range_rejected(tmp_path):
    make(tmp_path, 1000, "val")
    lab = tmp_path / "dataset" / "val_labels.bin"
    arr = np.fromfile(lab, np.uint8); arr[3] = 2; arr.tofile(lab)
    with pytest.raises(ValueError):
        Dataset(tmp_path, "val")


def test_sequential_batches_cover_split_in_order(tmp_path):
    ds = Dataset(make(tmp_path, 1000, "val"), "val")
    sizes = [len(y) for _, y in ds.sequential_batches(32)]
    assert sizes == [32] * 31 + [8]
    assert np.array_equal(np.concatenate([y for _, y in ds.sequential_batches(32)]), ds.labels)


def test_permutation_validation(tmp_path):
    (tmp_path / "permutations").mkdir()
    good = np.random.default_rng(0).permutation(8000).astype("<u4")
    good.tofile(tmp_path / "permutations" / "train_epoch_001.perm")
    assert np.array_equal(load_permutation(tmp_path, 1), good)
    bad = good.copy(); bad[0] = bad[1]
    bad.tofile(tmp_path / "permutations" / "train_epoch_002.perm")
    with pytest.raises(ValueError):
        load_permutation(tmp_path, 2)
    for e in (0, EPOCHS + 1):
        with pytest.raises(ValueError):
            load_permutation(tmp_path, e)


@needs_data
def test_real_dataset_loads_with_contract_sizes():
    for split, n, dogs in (("train", 8000, 4000), ("val", 1000, 500), ("test", 2000, 1000)):
        ds = Dataset(SHARED, split)
        assert len(ds) == n and int(ds.labels.sum()) == dogs
    assert all(load_permutation(SHARED, e).size == 8000 for e in range(1, EPOCHS + 1))


@needs_data
def test_golden_real_batch_matches_loader():
    """train_step/case_002 (lote real 1 de la época 1) debe coincidir con lo que produce el cargador del motor."""
    from src.core.tensor_utils import read_tb
    t = read_tb(SHARED / "test_vectors" / "train_step" / "case_002.tb")
    x, y = Dataset(SHARED, "train").batch(load_permutation(SHARED, 1)[:32])
    assert np.array_equal(x, t["input"]) and np.array_equal(y, t["labels"])
