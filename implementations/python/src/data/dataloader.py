"""Lectura de los artefactos compartidos (docs/dataset.md §6-§8): ficheros enteros a RAM, sin memmap (D-032)."""
from pathlib import Path

import numpy as np

from .preprocessing import to_float32

SIZES = {"train": 8000, "val": 1000, "test": 2000}
EPOCHS = 20


class Dataset:
    """Un conjunto (train/val/test): imágenes uint8 (N, 3072) y etiquetas uint8 (N,) leídas íntegras a memoria."""

    def __init__(self, data_dir, split):
        d = Path(data_dir) / "dataset"
        n = SIZES[split]
        images = np.fromfile(d / f"{split}_images.bin", dtype=np.uint8)
        labels = np.fromfile(d / f"{split}_labels.bin", dtype=np.uint8)
        if images.size != n * 3072 or labels.size != n:
            raise ValueError(f"{split}: tamaños {images.size} / {labels.size}, se esperaban {n * 3072} / {n}")
        if labels.max() > 1:
            raise ValueError(f"{split}: etiquetas fuera de {{0, 1}}")
        self.split, self.images, self.labels = split, images.reshape(n, 3072), labels

    def __len__(self):
        return len(self.labels)

    def batch(self, positions):
        """positions: posiciones del fichero → (X float32 (B,3,32,32), Y uint8 (B,)). Normaliza aquí, por lote."""
        positions = np.asarray(positions)
        if positions.size == 0 or positions.min() < 0 or positions.max() >= len(self):
            raise IndexError(f"posiciones fuera de [0, {len(self)})")
        return to_float32(self.images[positions]), self.labels[positions]

    def sequential_batches(self, size=32):
        """Orden de manifiesto sin barajar (validación/test): 31×32+8 en val, 62×32+16 en test."""
        for start in range(0, len(self), size):
            yield self.batch(np.arange(start, min(start + size, len(self))))


def load_permutation(data_dir, epoch):
    """Permutación compartida de la época `epoch` (1…20): 8000 uint32 LE, validada como permutación exacta."""
    if not 1 <= epoch <= EPOCHS:
        raise ValueError(f"época {epoch} fuera de [1, {EPOCHS}]")
    perm = np.fromfile(Path(data_dir) / "permutations" / f"train_epoch_{epoch:03d}.perm", dtype="<u4")
    if perm.size != SIZES["train"] or not np.array_equal(np.sort(perm), np.arange(SIZES["train"], dtype=np.uint32)):
        raise ValueError(f"época {epoch}: el fichero no es una permutación de 0…7999")
    return perm
