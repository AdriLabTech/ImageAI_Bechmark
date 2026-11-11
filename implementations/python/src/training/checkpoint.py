"""Checkpoints TensorBundle con metadatos `meta.*` escalares (docs/architecture.md §5.5, training.md §6)."""
import shutil
from pathlib import Path

import numpy as np

from ..core.tensor_utils import read_tb

CONTRACT_VERSION = 1   # contrato v1.0 (architecture.md §11); mismo valor que manifest.csv de los vectores


def meta_tensors(epoch, train_loss, val_loss, val_accuracy):
    return {
        "meta.epoch": np.array([epoch], dtype="<u4"),
        "meta.train_loss": np.array([train_loss], dtype="<f4"),
        "meta.val_loss": np.array([val_loss], dtype="<f4"),
        "meta.val_accuracy": np.array([val_accuracy], dtype="<f4"),
        "meta.contract_version": np.array([CONTRACT_VERSION], dtype="<u4"),
    }


def save_checkpoint(model, out_dir, epoch, train_loss, val_loss, val_accuracy):
    """Escribe checkpoint_epoch_NNN.tb y lo copia a checkpoint_latest.tb. Devuelve la ruta del primero."""
    out_dir = Path(out_dir)
    path = out_dir / f"checkpoint_epoch_{epoch:03d}.tb"
    model.save(path, meta_tensors(epoch, train_loss, val_loss, val_accuracy))
    shutil.copyfile(path, out_dir / "checkpoint_latest.tb")
    return path


def load_checkpoint_meta(path):
    t = read_tb(path)
    return {k[5:]: v[0].item() for k, v in t.items() if k.startswith("meta.")}
