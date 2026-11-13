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
    def __init__(self, data_dir, out_dir, run_id="run", epochs=EPOCHS, initial_weights=None, log=print, warmup=False):
        self.data_dir, self.out_dir, self.run_id, self.epochs, self.log = Path(data_dir), Path(out_dir), run_id, epochs, log
        self.warmup = warmup          # benchmark.md §5.2: 1 época sobre una copia y recarga de los pesos iniciales (D-031)
        self.perf = {}                # tiempos del motor (benchmark.md §5.3); se rellena en train()
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

    def _epoch(self, model, opt, train, perm, batch_times=None):
        """Una época de training.md §3. Devuelve la lista de pérdidas de lote."""
        params, losses = model.params(), []
        for b in range(len(train) // BATCH):
            t0 = time.perf_counter_ns()
            x, y = train.batch(perm[b * BATCH:(b + 1) * BATCH])
            z = model.forward(x)
            loss, dz = bce_with_logits(z, y)              # lanza FloatingPointError si NaN/Inf
            model.backward(dz)
            opt.step(params, model.grads())
            if batch_times is not None:
                batch_times.append((time.perf_counter_ns() - t0) / 1e9)
            losses.append(loss)
        return losses

    def train(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()                      # data_load_s (D-032): todo a RAM antes de cronometrar el entrenamiento
        train, val, test = (Dataset(self.data_dir, s) for s in ("train", "val", "test"))
        perms = [load_permutation(self.data_dir, e) for e in range(1, EPOCHS + 1)]
        data_load_s = time.perf_counter() - t0
        warmup_s = 0.0
        if self.warmup:                               # sobre una COPIA: el modelo medido se carga después, desde el fichero
            t0 = time.perf_counter()
            wm = self._load_model()
            self._epoch(wm, SGD(), train, perms[0])
            evaluate(wm, val)
            warmup_s = time.perf_counter() - t0
            del wm
        model, opt = self._load_model(), SGD()
        log_rows, batch_times, epoch_times, val_times = [], [], [], []
        for e in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            batch_losses = self._epoch(model, opt, train, perms[e - 1], batch_times)
            epoch_time = time.perf_counter() - t0
            train_loss = float(_mean32(batch_losses))
            t0 = time.perf_counter()
            val_loss, val_acc, _ = evaluate(model, val)
            val_times.append(time.perf_counter() - t0)
            epoch_times.append(epoch_time)
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
        self.perf = {"engine": ENGINE, "run": self.run_id, "epochs": self.epochs, "warmup": self.warmup,
                     "data_load_s": round(data_load_s, 6), "warmup_s": round(warmup_s, 6),
                     "train_total_s": round(sum(epoch_times), 6),
                     "epoch_time_s": [round(t, 6) for t in epoch_times],
                     "val_epoch_s": [round(t, 6) for t in val_times],
                     "batch_time_s": [round(t, 6) for t in batch_times]}
        return self._final(model, val, test, log_rows)

    def _final(self, model, val, test, log_rows):
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
