#!/usr/bin/env bash
# Medición oficial del motor Python (docs/benchmark.md): 5 ejecuciones completas + referencia de Python.
# Uso:  bash scripts/medicion_oficial_python.sh [--yes]
# No sobrescribe nada: si ya existen resultados o referencia, se detiene.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
ENGINE="$REPO/implementations/python"
RESULTS="$REPO/benchmark/results/python"
REF="$REPO/reports/reference/python_reference.json"
fail() { echo "✗ $*" >&2; exit 1; }

echo "== Comprobaciones previas =="
for f in train test val; do
  [ -f "shared/dataset/${f}_images.bin" ] || fail "falta shared/dataset/${f}_images.bin (ver README: scripts/prepare_dataset.py)"
done
[ -f shared/permutations/train_epoch_020.perm ] || fail "faltan las permutaciones (scripts/prepare_dataset.py)"
[ -x "$ENGINE/.venv/bin/python" ] || fail "falta el entorno virtual: cd implementations/python && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
command -v taskset >/dev/null || fail "falta taskset (paquete util-linux)"
if [ -d "$RESULTS" ] && [ -n "$(ls -A "$RESULTS" 2>/dev/null | grep -v '^.gitkeep$' || true)" ]; then
  fail "ya hay resultados en $RESULTS; muévelos o bórralos tú a mano antes de repetir"
fi
[ ! -e "$REF" ] || fail "ya existe $REF; la referencia no se sobrescribe sin que lo decidas tú"
if pgrep -f "src\.main (train|bench-infer|tests)" >/dev/null; then fail "hay otro entrenamiento/medición en marcha"; fi

GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo desconocido)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "  governor de CPU : $GOV  (recomendado: performance)"
echo "  carga del equipo: $LOAD (1 min; recomendado < 1,0)"
[ "$GOV" = "performance" ] || echo "  ⚠ el governor no es 'performance': se registrará tal cual en session.json"
if awk "BEGIN{exit !($LOAD >= 1.0)}"; then
  echo "  ⚠ el equipo no está en reposo; cierra navegador, vídeo, compilaciones... y vuelve a lanzar" >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
  echo "  FORCE=1: se sigue igualmente y la carga queda registrada en session.json"
fi
echo "  alimentación    : conecta el cargador (mide con corriente alterna)"
echo
echo "Va a tardar unos 8-9 minutos. No uses el equipo mientras tanto."
if [ "${1:-}" != "--yes" ]; then read -r -p "¿Empezar? [s/N] " ok; [ "$ok" = "s" ] || [ "$ok" = "S" ] || exit 0; fi

PYV=$("$ENGINE/.venv/bin/python" -c 'import platform; print(platform.python_version())')
NPV=$("$ENGINE/.venv/bin/python" -c 'import numpy; print(numpy.__version__)')
LOG="$(mktemp -d)/sesion.log"
set +e
python3 benchmark/harness/run_benchmark.py --engine python --cwd "$ENGINE" --cmd ".venv/bin/python -m src.main" \
  --exe-path "$ENGINE/.venv" --src "$ENGINE/src" --tests "$ENGINE/tests" --ext .py \
  --runtime-deps "numpy==$NPV" --build-commands 3 --make-reference \
  --meta "language_version=$PYV" --meta "numpy_version=$NPV" --notes "${NOTAS:-}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
[ -d "$RESULTS" ] && cp "$LOG" "$RESULTS/sesion.log" || true
echo
if [ "$RC" -eq 0 ]; then
  echo "✓ Medición terminada. Resultados: benchmark/results/python/summary.json y reports/reference/"
  python3 - <<'PY'
import json
s = json.load(open("benchmark/results/python/summary.json"))
p, q, r, v = s["performance"], s["quality"], s["resources"], s["validity"]
print(f"  entrenamiento (mediana de 5): {p['train_total_s']['median']:.1f} s  (variación {p['train_total_s']['rel_std']*100:.1f} %)")
print(f"  inferencia por imagen       : {p['inference_latency_s']['median']*1e3:.3f} ms;  {p['inference_throughput_img_s']['median']:.0f} imágenes/s en lotes de 32")
print(f"  memoria máxima              : {r['ram_max_mb']:.0f} MB")
print(f"  acierto en test             : {q['accuracy']*100:.2f} %   (AUC {q['auc_roc']:.3f})")
print(f"  determinista: {v['deterministic']}   atípicos: {v['any_outlier']}   repetir sesión (>5 % de variación): {v['needs_repeat_rel_std_gt_5pct']}")
PY
else
  echo "✗ La medición terminó con error (código $RC). Log: $LOG" >&2
fi
exit "$RC"
