"""Ciclo de entrenamiento exacto de docs/training.md §3-§7. Sin azar en runtime: todo viene de artefactos compartidos."""
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..core.initializers import load_initial_weights
from ..core.losses import bce_per_sample, bce_with_logits
from ..data.dataloader import EPOCHS, Dataset, load_permutation
from ..model.cnn import CNN
from .checkpoint import save_checkpoint
from .optimizer import SGD

BATCH = 32
ENGINE = "python"


def _mean32(values):
    """Media en FP32 con suma secuencial en el orden dado (cumsum no reordena)."""
    v = np.asarray(values, dtype=np.float32)
    return np.cumsum(v, dtype=np.float32)[-1] / np.float32(v.size)


def evaluate(model, dataset):
    """§4: media de pérdida sobre TODO el conjunto y exactitud por logit ≥ 0. Devuelve (loss, acc, logits (N,))."""
    losses, logits = [], []
    for x, y in dataset.sequential_batches(BATCH):
        z = model.forward(x)
        losses.append(bce_per_sample(z, y.reshape(z.shape).astype(np.float32)).reshape(-1))
        logits.append(z.reshape(-1).copy())
    logits = np.concatenate(logits)
    loss = _mean32(np.concatenate(losses))
    if not np.isfinite(loss):
        raise FloatingPointError(f"{dataset.split}: pérdida no finita")
    acc = np.float32(np.count_nonzero((logits >= 0) == (dataset.labels == 1))) / np.float32(len(dataset))
    return float(loss), float(acc), logits


class Trainer:
    def __init__(self, data_dir, out_dir, run_id="run", epochs=EPOCHS, initial_weights=None, log=print):
        self.data_dir, self.out_dir, self.run_id, self.epochs, self.log = Path(data_dir), Path(out_dir), run_id, epochs, log
        if not 1 <= epochs <= EPOCHS:
            raise ValueError(f"epochs debe estar en [1, {EPOCHS}]")
        self.initial_weights = Path(initial_weights) if initial_weights else self.data_dir / "initial_weights" / "initial_weights.tb"

    def _load_model(self):
        model = CNN()
        model.load_params(load_initial_weights(self.initial_weights))
        # criterio 5 (§10): los pesos cargados son bit-idénticos a los del fichero compartido
        ref = load_initial_weights(self.initial_weights)
        for name, arr in model.params().items():
            if arr.tobytes() != ref[name].tobytes():
                raise RuntimeError(f"{name}: los pesos cargados difieren de initial_weights.tb")
        return model

    def train(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        train, val = Dataset(self.data_dir, "train"), Dataset(self.data_dir, "val")
        model, opt = self._load_model(), SGD()
        params = model.params()
        log_rows = []
        for e in range(1, self.epochs + 1):
            perm = load_permutation(self.data_dir, e)
            batch_losses = []
            t0 = time.perf_counter()
            for b in range(len(train) // BATCH):
                x, y = train.batch(perm[b * BATCH:(b + 1) * BATCH])
                z = model.forward(x)
                loss, dz = bce_with_logits(z, y)          # lanza FloatingPointError si NaN/Inf
                model.backward(dz)
                opt.step(params, model.grads())
                batch_losses.append(loss)
            epoch_time = time.perf_counter() - t0
            train_loss = float(_mean32(batch_losses))
            val_loss, val_acc, _ = evaluate(model, val)
            save_checkpoint(model, self.out_dir, e, train_loss, val_loss, val_acc)
            row = {"engine": ENGINE, "run": self.run_id, "epoch": e, "train_loss": train_loss,
                   "val_loss": val_loss, "val_accuracy": val_acc, "epoch_time_s": round(epoch_time, 3)}
            log_rows.append(row)
            self.log(f"época {e:2d}/{self.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                     f"val_acc={val_acc:.4f}  {epoch_time:.1f}s")
        with open(self.out_dir / "train_log.jsonl", "w", encoding="utf-8") as f:
            for row in log_rows:
                f.write(json.dumps(row) + "\n")
        shutil.copyfile(self.out_dir / f"checkpoint_epoch_{self.epochs:03d}.tb", self.out_dir / "model_final.tb")
        return self._final(model, val, log_rows)

    def _final(self, model, val, log_rows):
        test = Dataset(self.data_dir, "test")
        val_loss, val_acc, _ = evaluate(model, val)
        test_loss, test_acc, logits = evaluate(model, test)
        with open(self.out_dir / "test_logits.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["position", "logit"])
            for i, z in enumerate(logits):
                w.writerow([i, "%.9g" % z])
        metrics = {"engine": ENGINE, "run": self.run_id, "epochs": self.epochs, "val_loss": val_loss,
                   "val_accuracy": val_acc, "test_loss": test_loss, "test_accuracy": test_acc}
        (self.out_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics
