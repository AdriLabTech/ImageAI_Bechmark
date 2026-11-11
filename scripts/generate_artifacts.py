"""Genera los artefactos compartidos de la FASE 2: pesos iniciales y vectores dorados de operaciones.

Uso: implementations/python/.venv/bin/python scripts/generate_artifacts.py
Los vectores se calculan en FP64 con las operaciones del motor (validadas por gradcheck) sobre entradas FP32
exactas y se guardan como FP32 (architecture.md §10.2). Sin numpy.random: todo sale de splitmix64.
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "implementations" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prng import SEED, SplitMix64                                  # noqa: E402
from src.core.activations import relu_backward, relu_forward, sigmoid_backward, sigmoid_forward  # noqa: E402
from src.core.losses import bce_with_logits                         # noqa: E402
from src.core.tensor_utils import write_tb                          # noqa: E402
from src.layers.conv2d import Conv2D                                # noqa: E402
from src.layers.dense import Dense                                  # noqa: E402
from src.layers.flatten import Flatten                              # noqa: E402
from src.layers.maxpool2d import MaxPool2D                          # noqa: E402
from src.model.cnn import CNN                                       # noqa: E402
from src.training.optimizer import LEARNING_RATE                    # noqa: E402

CONTRACT_VERSION = 1
WEIGHTS_SEED = SEED + 1000          # architecture.md §7.1
VECTORS_SEED = SEED + 2000          # +k para el caso k-ésimo (orden de generación)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- pesos iniciales
def initial_weights():
    rng = SplitMix64(WEIGHTS_SEED)
    spec = [("conv1.weight", (16, 3, 3, 3), 2 / 27), ("conv2.weight", (32, 16, 3, 3), 2 / 144),
            ("dense1.weight", (64, 1152), 2 / 1152), ("dense2.weight", (1, 64), 2 / 65)]   # std² (He / Xavier)
    out = {}
    for name, shape, var in spec:
        n = int(np.prod(shape))
        assert n % 2 == 0
        z = np.empty(n)
        for i in range(0, n, 2):
            z[i], z[i + 1] = rng.normal_pair()
        out[name] = (np.sqrt(var) * z).astype(np.float32).reshape(shape)
    for name, size in [("conv1.bias", 16), ("conv2.bias", 32), ("dense1.bias", 64), ("dense2.bias", 1)]:
        out[name] = np.zeros(size, np.float32)
    return out


# ---------------------------------------------------------------- vectores dorados
class Gen:
    def __init__(self, k):
        self.rng = SplitMix64(VECTORS_SEED + k)

    def uniform(self, shape, lo=-1.0, hi=1.0):
        n = int(np.prod(shape))
        u = np.array([self.rng.uniform() for _ in range(n)])
        return (lo + (hi - lo) * u).astype(np.float32).reshape(shape)


    def images(self, n):
        """n imágenes sintéticas deterministas: uint8 de splitmix64 normalizado como el contrato (float32(u8)/255.0)."""
        u8 = np.array([self.rng.next() & 0xFF for _ in range(n * 3072)], np.uint8).reshape(n, 3, 32, 32)
        return u8.astype(np.float32) / np.float32(255.0)


f32 = lambda a: np.asarray(a, dtype=np.float64).astype(np.float32)
f64 = lambda a: np.asarray(a, dtype=np.float64)


def conv_case(g, n, c, o, hw, xr, wr, br):
    x, w, b = g.uniform((n, c, hw, hw), *xr), g.uniform((o, c, 3, 3), *wr), g.uniform((o,), *br)
    layer = Conv2D(c, o, 3, np.float64)
    layer.set_params(f64(w), f64(b))
    y = layer.forward(f64(x))
    dy = g.uniform(y.shape)
    dx = layer.backward(f64(dy))
    return dict(input=x, weight=w, bias=b, output=f32(y), grad_output=dy, grad_input=f32(dx),
                grad_weight=f32(layer.grad_weight), grad_bias=f32(layer.grad_bias))


def pool_case(g, x, dy=None):
    layer = MaxPool2D()
    y = layer.forward(f64(x))
    dy = g.uniform(y.shape) if dy is None else dy
    return dict(input=x, output=f32(y), grad_output=dy, grad_input=f32(layer.backward(f64(dy))))


def assert_no_ties(x):
    n, c, h, w = x.shape
    win = x[:, :, : h // 2 * 2, : w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
    top2 = np.sort(win, axis=-1)[..., -2:]
    assert (top2[..., 1] > top2[..., 0]).all(), "caso sin empates con empate accidental"


def build_cases(weights):
    cases = []                                          # (operación, purpose, tensores)
    k = [0]

    def gen():
        k[0] += 1
        return Gen(k[0])

    cases += [("conv2d", "formas reales de conv1 (1,3,32,32)", conv_case(gen(), 1, 3, 16, 32, (0, 1), (-.3, .3), (-.1, .1))),
              ("conv2d", "formas reales de conv2 (1,16,15,15)", conv_case(gen(), 1, 16, 32, 15, (0, 1), (-.1, .1), (-.1, .1))),
              ("conv2d", "pequeño para depurar a mano: N=2, C=1, 5×5, O=2", conv_case(gen(), 2, 1, 2, 5, (-1, 1), (-1, 1), (-1, 1))),
              ("conv2d", "entrada y bias negativos (2,3,8,8)", conv_case(gen(), 2, 3, 4, 8, (-2, 0), (-1, 1), (-1, 0)))]

    for shape, purpose in [((1, 16, 30, 30), "entrada par 30×30 (formas reales de pool1)"),
                           ((2, 4, 13, 13), "entrada impar 13×13: descarta fila y columna 12")]:
        g = gen()
        x = g.uniform(shape)
        assert_no_ties(x)
        cases.append(("maxpool2d", purpose, pool_case(g, x)))
    g = gen()                                            # empates: la regla del primer máximo en orden raster
    x = np.zeros((1, 1, 4, 4), np.float32)
    x[0, 0, 0:2, 0:2] = [[5, 5], [5, 5]]
    x[0, 0, 0:2, 2:4] = [[1, 7], [7, 2]]
    x[0, 0, 2:4, 0:2] = [[3, 3], [9, 9]]
    cases.append(("maxpool2d", "empates de máximo (regla del primer máximo en orden raster)",
                  pool_case(g, x, np.array([1, 2, 3, 4], np.float32).reshape(1, 1, 2, 2))))
    g = gen()
    x = g.uniform((1, 32, 13, 13))
    assert_no_ties(x)
    cases.append(("maxpool2d", "formas reales de pool2 (1,32,13,13)", pool_case(g, x)))

    g = gen()
    x = np.array([[-2, -1, -0.0, 0.0], [0.0, 0.5, 1, 3]], np.float32)
    dy = g.uniform(x.shape)
    cases.append(("relu", "ceros exactos y negativos; derivada en 0 = 0",
                  dict(input=x, output=f32(relu_forward(f64(x))), grad_output=dy, grad_input=f32(relu_backward(f64(x), f64(dy))))))
    g = gen()
    x = g.uniform((1, 16, 30, 30)) * np.tile(np.array([1e-30, 1.0, 1e6], np.float32), 4800)[:14400].reshape(1, 16, 30, 30)
    x = x.astype(np.float32)
    dy = g.uniform(x.shape)
    cases.append(("relu", "formas reales con magnitudes extremas (1e-30, 1, 1e6)",
                  dict(input=x, output=f32(relu_forward(f64(x))), grad_output=dy, grad_input=f32(relu_backward(f64(x), f64(dy))))))

    for purpose, x in [("rango [-30, 30] (estabilidad)", np.linspace(-30, 30, 64).astype(np.float32).reshape(4, 16)),
                       ("rango fino alrededor de 0", gen().uniform((4, 16), -1e-3, 1e-3))]:
        dy = gen().uniform(x.shape)
        s = sigmoid_forward(f64(x))
        cases.append(("sigmoid", purpose, dict(input=x, output=f32(s), grad_output=dy, grad_input=f32(sigmoid_backward(s, f64(dy))))))

    g = gen()
    x = g.uniform((2, 32, 6, 6))
    layer = Flatten()
    y = layer.forward(x)
    dy = g.uniform(y.shape)
    cases.append(("flatten", "forma real (2,32,6,6) → (2,1152)", dict(input=x, output=y.copy(), grad_output=dy, grad_input=layer.backward(dy))))

    for n, i, o, purpose, xr, wr in [(4, 1152, 64, "formas reales de dense1", (0, 1), (-.05, .05)),
                                      (3, 4, 4, "cuadrado pequeño verificable a mano", (-1, 1), (-1, 1))]:
        g = gen()
        x, w, b = g.uniform((n, i), *xr), g.uniform((o, i), *wr), g.uniform((o,), -.1, .1)
        layer = Dense(i, o, np.float64)
        layer.set_params(f64(w), f64(b))
        y = layer.forward(f64(x))
        dy = g.uniform(y.shape)
        dx = layer.backward(f64(dy))
        cases.append(("dense", purpose, dict(input=x, weight=w, bias=b, output=f32(y), grad_output=dy, grad_input=f32(dx),
                                              grad_weight=f32(layer.grad_weight), grad_bias=f32(layer.grad_bias))))

    for purpose, z, y in [("logits = 0", np.zeros((8, 1), np.float32), np.array([0, 1] * 4, np.uint8)),
                          ("logits extremos sin desbordamiento", np.array([-100, -30, -1, 0, 1, 30, 100, 500], np.float32).reshape(8, 1),
                           np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint8)),
                          ("mezcla realista (32,1)", None, None)]:
        if z is None:
            g = gen()
            z = g.uniform((32, 1), -4, 4)
            y = np.array([int(g.rng.next() & 1) for _ in range(32)], np.uint8)
        loss, dz = bce_with_logits(f64(z), y)
        cases.append(("bce", purpose, dict(input=z, labels=y, loss=f32([loss]), grad_input=f32(dz))))

    # ---- modelo completo (FASE 3): batch de 4 imágenes sintéticas con los pesos iniciales compartidos
    g = gen()
    x = g.images(4)
    model = CNN(np.float64)
    model.load_params({k: f64(v) for k, v in weights.items()})
    z = model.forward(f64(x))
    t = dict(input=x, logits=f32(z))
    for i, a in enumerate(model.outputs[:9], start=1):          # a1…a9 (model.md §5)
        t[f"a{i}"] = f32(a)
    cases.append(("model", "forward completo, 4 imágenes sintéticas, pesos = initial_weights.tb", t))

    # ---- T2: un paso de entrenamiento con un lote sintético de 32 (el lote real 1 llega en la FASE 4, D-034)
    g = gen()
    x = g.images(32)
    y = np.array([int(g.rng.next() & 1) for _ in range(32)], np.uint8)
    model = CNN(np.float64)
    model.load_params({k: f64(v) for k, v in weights.items()})
    loss, dz = bce_with_logits(model.forward(f64(x)), y)
    model.backward(dz)
    grads, params = model.grads(), model.params()
    t = dict(input=x, labels=y, loss=f32([loss]))
    for name in sorted(weights):
        t[f"weights_init.{name}"] = weights[name]
        t[f"grads.{name}"] = f32(grads[name])
        t[f"weights_updated.{name}"] = f32(f64(weights[name]) - LEARNING_RATE * grads[name])
    cases.append(("train_step", "un paso SGD (lr 0,01), lote SINTÉTICO de 32 con pesos iniciales", t))
    return cases


def main():
    wdir = ROOT / "shared" / "initial_weights"
    wdir.mkdir(parents=True, exist_ok=True)
    w = initial_weights()
    write_tb(wdir / "initial_weights.tb", w)
    (wdir / "CHECKSUMS.sha256").write_text(f"{sha256(wdir / 'initial_weights.tb')}  initial_weights.tb\n")
    print("pesos iniciales:", sum(v.size for v in w.values()), "parámetros")

    tdir = ROOT / "shared" / "test_vectors"
    counters, rows = {}, []
    for op, purpose, tensors in build_cases(w):
        counters[op] = counters.get(op, 0) + 1
        case = f"case_{counters[op]:03d}"
        (tdir / op).mkdir(parents=True, exist_ok=True)
        path = tdir / op / f"{case}.tb"
        write_tb(path, tensors)
        rows.append([f"{op}/{case}", op, purpose, sha256(path), CONTRACT_VERSION])
    with open(tdir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["case", "operation", "purpose", "sha256", "contract_version"])
        wr.writerows(rows)
    print(len(rows), "vectores dorados escritos en", tdir)


if __name__ == "__main__":
    main()
