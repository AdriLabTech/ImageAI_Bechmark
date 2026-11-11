"""Comprobaciones aritméticas de la especificación (FASE 1). Solo biblioteca estándar.

Uso: python3 -I tests/check_spec.py   (exit 0 si todo pasa)
"""
import math, re, statistics, sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
fails = 0
KNOWN = ()  # defectos documentados y pendientes de decisión; no bloquean
def check(name, cond, extra=""):
    global fails
    known = (not cond) and name.startswith(KNOWN)
    print(("OK   " if cond else ("KNOWN" if known else "FAIL ")) + name + (f"  [{extra}]" if extra else ""))
    fails += (not cond) and not known

# --- Dimensiones y parámetros (model.md §3) ---
h = 32; h1 = h - 3 + 1; p1 = h1 // 2; h2 = p1 - 3 + 1; p2 = h2 // 2
check("dimensiones 30/15/13/6", (h1, p1, h2, p2) == (30, 15, 13, 6))
check("flatten 1152", 32 * p2 * p2 == 1152)
params = 16*3*9 + 16 + 32*16*9 + 32 + 64*1152 + 64 + 64 + 1
check("parámetros 78.945", params == 78945, str(params))
act = 32 * (3*32*32 + 2*16*30*30 + 16*15*15 + 2*32*13*13 + 32*6*6 + 2*64)  # X,A1,A2,A3,A4,A5,A6,A8,A9
check("activaciones conservadas (N=32) 1.522.176", act == 1522176, str(act))

# --- Inicialización (model.md §7) ---
check("std conv1/conv2/dense1/dense2",
      all(abs(a - b) < 5e-7 for a, b in [(math.sqrt(2/27), .272166), (math.sqrt(2/144), .117851),
                                         (math.sqrt(2/1152), 1/24), (math.sqrt(2/65), .175412)]))
check("tensores con nº par de elementos (Box-Muller sin sobrante)", all(n % 2 == 0 for n in (432, 4608, 73728, 64)))

# --- Cuotas (dataset.md §4) ---
neg = [0, 1, 2, 3, 4, 6, 7, 8, 9]
def quotas(total):
    base, r = divmod(total, 9)
    return {c: base + (1 if i < r else 0) for i, c in enumerate(neg)}
txt = (DOCS / "dataset.md").read_text(encoding="utf-8")
for col, total in ((1, 4000), (2, 500), (3, 1000)):
    q = quotas(total)
    check(f"cuotas calculadas suman {total}", sum(q.values()) == total)
    for c in neg:
        m = re.search(rf"^\| {c} \w+ \| (\d+) \| (\d+) \| (\d+) \|", txt, re.M)
        check(f"tabla dataset.md clase {c} col {col}", int(m.group(col)) == q[c], f"doc={m.group(col)} calc={q[c]}")
check("máx. necesario ≤ disponible (train 501 ≤ 5000; test 112 ≤ 1000)", 445 + 56 <= 5000 and 112 <= 1000)

# --- Lotes (training.md) ---
check("lotes train 250", 8000 // 32 == 250 and 8000 % 32 == 0)
check("lotes val 31+8", (1000 // 32, 1000 % 32) == (31, 8))
check("lotes test 62+16", (2000 // 32, 2000 % 32) == (62, 16))

# --- Tamaños de fichero (dataset.md §6) ---
check("dataset derivado ≈ 33,8 MB en disco", 11000*3072 + 11000 == 33_803_000, "imágenes 33.792.000 B + etiquetas 11.000 B")
check("train_images 24.576.000", 8000 * 3072 == 24_576_000)
check("RAM FP32 train ≈ 98,3 MB (93,75 MiB)", 8000*3072*4 == 98_304_000)

# --- TensorBundle (architecture.md §5) ---
entry = 32 + 2 + 1 + 1 + 4 + 8 + 8 + 16
check("entrada de índice = 72 B", entry == 72, str(entry))
check("shape 16 B caben 4 dimensiones uint32 (ndim ≤ 4)", 16 // 4 == 4)
check("ndim ≤ 4 en architecture.md", "∈ [1, 4]" in (DOCS / "architecture.md").read_text(encoding="utf-8"))
def tb_size(n, data): 
    pad = lambda b: (b + 3) // 4 * 4
    return 12 + 72 * n + sum(pad(d) for d in data)
w = [432*4, 16*4, 4608*4, 32*4, 73728*4, 64*4, 64*4, 1*4]
check("initial_weights.tb = 316.368 B", tb_size(8, w) == 316_368, str(tb_size(8, w)))
ck = tb_size(13, w + [4]*5)
check("model_final.tb = 316.748 B ≈ 0,3167 MB", ck == 316_748 and round(ck/1e6, 4) == 0.3167, str(ck))
check("nombres de tensor ≤ 31 caracteres", max(len(n) for n in ("weights_updated.dense1.weight", "weights_init.dense1.weight", "meta.contract_version")) <= 31)

# --- Regla de valores atípicos (D-030): la regla 2σ era inaplicable con n=5 ---
bench = (DOCS / "benchmark.md").read_text(encoding="utf-8")
check("benchmark.md usa regla MAD y no «2σ»", "MAD" in bench and "más de 2σ" not in bench)
xs = [10.0, 10.1, 9.9, 10.0, 12.0]
med = statistics.median(xs); mad = statistics.median(abs(x - med) for x in xs)
check("regla MAD marca el atípico con n=5", [x for x in xs if abs(x - med) > 3 * 1.4826 * mad] == [12.0])

print("\nFALLOS:", fails)
sys.exit(1 if fails else 0)
