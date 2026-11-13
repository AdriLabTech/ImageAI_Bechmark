"""Arnés de benchmark externo (docs/benchmark.md). Mide a un motor SOLO desde fuera: subproceso + /proc + CLI del contrato.

Uso (Python):
  python benchmark/harness/run_benchmark.py --engine python --cwd implementations/python \
      --cmd ".venv/bin/python -m src.main" --exe-path implementations/python/.venv --src implementations/python/src \
      --tests implementations/python/tests --ext .py --runtime-deps "numpy==X" --build-commands 3 --make-reference \
      --meta language_version=... --meta numpy_version=...

Pasos: corrección (`tests`, sin cronómetro) → 5 ejecuciones de entrenamiento con calentamiento y muestreo de /proc cada 100 ms
→ inferencia (latencia y throughput) tras cada una → calidad con scripts/evaluate.py → summary.json y session.json.
Un motor que no pasa la corrección o no es determinista NO se mide (benchmark.md §2, §7).
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate  # noqa: E402

ENV_THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
SAMPLE_S = 0.1
CLK = os.sysconf("SC_CLK_TCK")


# ---------- utilidades estadísticas -------------------------------------------------
def pct(values, q):
    """Percentil q (0-100) por interpolación lineal (igual que numpy.percentile 'linear')."""
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def describe(values):
    return {"median": st.median(values), "mean": st.fmean(values), "std": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n": len(values)}


def outliers(values):
    """benchmark.md §5.2 (D-030): |x−mediana| > 3·1,4826·MAD o desviación > 10 % de la mediana. No elimina nada."""
    med = st.median(values)
    mad = st.median(abs(v - med) for v in values)
    flags = []
    for v in values:
        by_mad = mad > 0 and abs(v - med) > 3 * 1.4826 * mad
        flags.append(bool(by_mad or abs(v - med) > 0.10 * med))
    return flags


# ---------- entorno -----------------------------------------------------------------
def read(p, default=None):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return default


def cpu_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Package id 0:\s*\+?([\d.]+)", out) or re.search(r"Tctl:\s*\+?([\d.]+)", out)
        return float(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def machine(core):
    cpuinfo = read("/proc/cpuinfo", "")
    model = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    mem = re.search(r"MemTotal:\s*(\d+)", read("/proc/meminfo", ""))
    gov = read(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
    freq = read(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq")
    return {"cpu_model": model.group(1) if model else platform.processor(), "cpu_logical": os.cpu_count(),
            "ram_gb": round(int(mem.group(1)) / 1048576, 1) if mem else None, "kernel": platform.release(),
            "governor": gov, "core": core, "core_freq_mhz_at_start": round(int(freq) / 1000) if freq else None,
            "loadavg_1_5_15": list(os.getloadavg())}


# ---------- ejecución de un subproceso medido ---------------------------------------
def run_measured(argv, cwd, core, log_path):
    """Lanza `taskset -c core <argv>` (taskset hace exec: el pid es el del motor), muestrea /proc cada 100 ms."""
    env = {**os.environ, **ENV_THREADS}
    cmd = ["taskset", "-c", str(core)] + argv
    samples, prev = [], None
    t_start = time.monotonic_ns()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                rss_kb = int(re.search(r"VmRSS:\s*(\d+)", Path(f"/proc/{p.pid}/status").read_text()).group(1))
                f = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
                ticks, now = int(f[11]) + int(f[12]), time.monotonic()      # utime + stime
                cpu = None
                if prev:
                    cpu = (ticks - prev[0]) / CLK / (now - prev[1]) * 100
                prev = (ticks, now)
                samples.append((rss_kb / 1024, cpu))
            except (OSError, AttributeError, IndexError):
                pass                                                         # el proceso terminó entre medias
            time.sleep(SAMPLE_S)
    wall = (time.monotonic_ns() - t_start) / 1e9
    rss = [s[0] for s in samples]
    cpus = [s[1] for s in samples if s[1] is not None]
    res = {"samples": len(samples), "sample_interval_s": SAMPLE_S,
           "ram_max_mb": round(max(rss), 3) if rss else None, "ram_mean_mb": round(st.fmean(rss), 3) if rss else None,
           "cpu_mean_pct": round(st.fmean(cpus), 2) if cpus else None, "cpu_max_pct": round(max(cpus), 2) if cpus else None}
    return p.returncode, wall, res


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return round(total / 1048576, 3)


def loc(path, ext):
    """Líneas no vacías que no son solo comentario `#` (docstrings cuentan como código: asimetría documentada)."""
    n = 0
    for f in Path(path).rglob(f"*{ext}"):
        if "__pycache__" in f.parts:
            continue
        n += sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--cmd", required=True, help="prefijo del ejecutable del motor, p. ej. '.venv/bin/python -m src.main'")
    ap.add_argument("--data-dir", default=str(REPO / "shared"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--exe-path")
    ap.add_argument("--src")
    ap.add_argument("--tests")
    ap.add_argument("--ext", default=".py")
    ap.add_argument("--runtime-deps", default="")
    ap.add_argument("--build-commands", type=int)
    ap.add_argument("--meta", action="append", default=[], help="clave=valor registrado en meta")
    ap.add_argument("--make-reference", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--results", help="por defecto benchmark/results/<engine>")
    a = ap.parse_args()

    results = Path(a.results) if a.results else REPO / "benchmark" / "results" / a.engine
    if results.exists() and any(p.name != ".gitkeep" for p in results.iterdir()):
        sys.exit(f"error: {results} ya contiene resultados; no se sobrescriben (usa otro --results)")
    results.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("BENCH_SCRATCH", "/tmp")) / f"bench_{a.engine}_{int(time.time())}"
    scratch.mkdir(parents=True)
    cwd, engine = Path(a.cwd).resolve(), shlex.split(a.cmd)
    labels = Path(a.data_dir) / "dataset" / "test_labels.bin"

    session = {"date": datetime.now().astimezone().isoformat(timespec="seconds"), "operator": "agente Claude Code, a petición de Adrián",
               "cpu_temp_c_before": cpu_temp(), "machine": machine(a.core), "notes": a.notes,
               "env": ENV_THREADS, "affinity": f"taskset -c {a.core}"}
    print(f"[sesión] carga previa {session['machine']['loadavg_1_5_15']}, T CPU {session['cpu_temp_c_before']} °C, "
          f"governor {session['machine']['governor']}", flush=True)

    # 1. Corrección previa, sin cronómetro (benchmark.md §5.1)
    print("[1/4] corrección: tests (T1 + T2 + determinismo de 1 época)...", flush=True)
    rc, _, _ = run_measured(engine + ["tests", "--data-dir", a.data_dir], cwd, a.core, scratch / "tests.log")
    if rc != 0:
        sys.exit(f"el motor NO pasa la corrección (código {rc}); no se mide. Ver {scratch / 'tests.log'}")

    # 2. Rendimiento: N ejecuciones completas con calentamiento
    perf_runs, shas, logits = [], [], []
    for i in range(1, a.runs + 1):
        rid, out = f"perf_run_{i:03d}", scratch / f"run_{i:03d}"
        print(f"[2/4] ejecución {i}/{a.runs}...", flush=True)
        perf_json = scratch / f"{rid}_engine.json"
        rc, wall, res_train = run_measured(
            engine + ["train", "--data-dir", a.data_dir, "--out", str(out.parent), "--run", out.name, "--warmup",
                      "--perf-out", str(perf_json)], cwd, a.core, scratch / f"{rid}_train.log")
        if rc != 0:
            sys.exit(f"el entrenamiento falló en la ejecución {i} (código {rc}); ver {scratch / (rid + '_train.log')}")
        inf_json = scratch / f"{rid}_infer.json"
        rc, wall_inf, res_inf = run_measured(engine + ["bench-infer", "--model", str(out / "model_final.tb"),
                                                       "--data-dir", a.data_dir, "--out", str(inf_json)],
                                             cwd, a.core, scratch / f"{rid}_infer.log")
        if rc != 0:
            sys.exit(f"bench-infer falló en la ejecución {i}")
        eng, inf = json.loads(perf_json.read_text()), json.loads(inf_json.read_text())
        shas.append(sha(out / "model_final.tb"))
        logits.append(out / "test_logits.csv")
        perf = {**eng, "wall_train_process_s": round(wall, 6), "inference": {
            **inf, "latency_median_s": st.median(inf["latency_s"]), "latency_p95_s": pct(inf["latency_s"], 95),
            "throughput_median_img_s": st.median(inf["throughput_img_s"]),
            "throughput_p95_img_s": pct(inf["throughput_img_s"], 95)}}
        (results / f"{rid}.json").write_text(json.dumps(perf) + "\n")
        (results / f"resources_run_{i:03d}.json").write_text(json.dumps({"train": res_train, "infer": res_inf}, indent=1) + "\n")
        perf_runs.append(perf)
        print(f"      train_total_s={perf['train_total_s']:.3f}  wall={wall:.2f}s  RAM máx={res_train['ram_max_mb']} MB  "
              f"lat. mediana={perf['inference']['latency_median_s'] * 1e3:.3f} ms", flush=True)
    session["cpu_temp_c_after"] = cpu_temp()
    session["loadavg_after"] = list(os.getloadavg())

    # 3. Determinismo entre ejecuciones completas y calidad
    deterministic = len(set(shas)) == 1
    run1 = scratch / "run_001"
    ref_dir = REPO / "reports" / "reference"
    ref_csv = ref_dir / f"{a.engine}_reference_test_logits.csv"
    quality = evaluate.evaluate(logits[0], labels)
    if a.make_reference:
        ref_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(logits[0], ref_csv)
        shutil.copyfile(run1 / "model_final.tb", ref_dir / f"{a.engine}_reference_model_final.tb")
        shutil.copyfile(run1 / "train_log.jsonl", ref_dir / f"{a.engine}_reference_train_log.jsonl")
        fm = json.loads((run1 / "final_metrics.json").read_text())
        (ref_dir / f"{a.engine}_reference.json").write_text(json.dumps({
            "engine": a.engine, "date": session["date"], "source": "perf_run_001 de la primera sesión validada",
            "model_final_sha256": shas[0], "test_logits_sha256": sha(ref_csv),
            "final_metrics_engine": fm, "quality_common_evaluator": quality, "train_log": "python_reference_train_log.jsonl"
            if a.engine == "python" else f"{a.engine}_reference_train_log.jsonl",
            "files": {"model": f"{a.engine}_reference_model_final.tb", "test_logits": ref_csv.name}}, indent=1) + "\n")
    if ref_csv.exists():
        quality["numeric_error_vs_reference"] = {
            "per_run": [evaluate.numeric_error(evaluate.read_logits(l), evaluate.read_logits(ref_csv)) for l in logits]}
        quality["numeric_error_vs_reference"].update(quality["numeric_error_vs_reference"]["per_run"][0])

    # 4. Agregación (benchmark.md §8)
    totals = [r["train_total_s"] for r in perf_runs]
    flags = outliers(totals)
    med_i = sorted(range(len(totals)), key=totals.__getitem__)[(len(totals) - 1) // 2]
    per_epoch = [st.median(r["epoch_time_s"][e] for r in perf_runs) for e in range(len(perf_runs[0]["epoch_time_s"]))]
    all_val = [v for r in perf_runs for v in r["val_epoch_s"]]
    res = [json.loads((results / f"resources_run_{i:03d}.json").read_text())["train"] for i in range(1, a.runs + 1)]
    rel_std = describe(totals)["std"] / st.median(totals)
    meta = dict(m.split("=", 1) for m in a.meta)
    model_mb = round(Path(run1 / "model_final.tb").stat().st_size / 1048576, 4)
    summary = {
        "meta": {"engine": a.engine, **meta, "cpu_model": session["machine"]["cpu_model"], "ram_gb": session["machine"]["ram_gb"],
                 "kernel": session["machine"]["kernel"], "governor": session["machine"]["governor"],
                 "affinity": f"core {a.core}", "blas_threads": 1, "date": session["date"], "build": "release", "fast_math": False,
                 "runs": a.runs, "warmup": "1 época sobre copia + recarga de pesos iniciales (D-031)"},
        "performance": {
            "data_load_s": describe([r["data_load_s"] for r in perf_runs]),
            "train_total_s": {**describe(totals), "per_run": totals, "outlier": flags, "rel_std": rel_std},
            "epoch_time_s": {"median": st.median(per_epoch), "per_epoch_median": per_epoch,
                             "std": st.stdev([t for r in perf_runs for t in r["epoch_time_s"]])},
            "batch_time_s": {"median": st.median(perf_runs[med_i]["batch_time_s"]),
                             "p95": pct(perf_runs[med_i]["batch_time_s"], 95), "n": len(perf_runs[med_i]["batch_time_s"]),
                             "from_run": med_i + 1},
            "val_epoch_s": {"median": st.median(all_val), "std": st.stdev(all_val)},
            "inference_latency_s": {"median": st.median(r["inference"]["latency_median_s"] for r in perf_runs),
                                    "p95": st.median(r["inference"]["latency_p95_s"] for r in perf_runs), "n": 1000,
                                    "note": "mediana entre ejecuciones de la mediana y del p95 de cada una (1000 medidas por ejecución)"},
            "inference_throughput_img_s": {"median": st.median(r["inference"]["throughput_median_img_s"] for r in perf_runs),
                                           "p95": st.median(r["inference"]["throughput_p95_img_s"] for r in perf_runs), "n": 100,
                                           "note": "el p95 es el percentil 95 del throughput (lotes rápidos); no es latencia"}},
        "resources": {"ram_max_mb": max(r["ram_max_mb"] for r in res), "ram_mean_mb": st.median(r["ram_mean_mb"] for r in res),
                      "cpu_mean_pct": st.median(r["cpu_mean_pct"] for r in res), "cpu_max_pct": max(r["cpu_max_pct"] for r in res),
                      "executable_size_mb": dir_size_mb(a.exe_path) if a.exe_path else None, "model_size_mb": model_mb},
        "quality": quality,
        "complexity": {"loc_src": loc(a.src, a.ext) if a.src else None,
                       "loc_total": (loc(a.src, a.ext) + loc(a.tests, a.ext)) if a.src and a.tests else None,
                       "dev_hours": None, "runtime_dependencies": a.runtime_deps or None,
                       "memory_mgmt": None, "debug_difficulty": None, "optimization_ease": None, "parallelization": None,
                       "note": "dev_hours y rúbricas 1-5 pendientes: sin registro fiable de horas; las valora Adrián (benchmark.md §6.4)"},
        "portability": {"build_commands": a.build_commands, "os_tested": ["linux"], "toolchain_size_mb": None,
                        "distribution": None, "ecosystem": None, "note": "rúbricas pendientes (benchmark.md §6.5)"},
        "validity": {"t1_pass": True, "t2_pass": True, "deterministic": deterministic, "t3_pass": None,
                     "determinism_basis": "SHA-256 de model_final.tb idéntico en las %d ejecuciones de la sesión" % a.runs,
                     "t3_note": "no evaluable hasta tener un segundo motor (training.md §9)",
                     "needs_repeat_rel_std_gt_5pct": rel_std > 0.05, "any_outlier": any(flags)},
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (results / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    # Artefactos de la ejecución 1 (pequeños): modelo, log y logits
    shutil.copyfile(run1 / "train_log.jsonl", results / "train_log_run_001.jsonl")
    shutil.copyfile(logits[0], results / "test_logits_run_001.csv")
    print(json.dumps({"train_total_s_median": summary["performance"]["train_total_s"]["median"], "rel_std": rel_std,
                      "deterministic": deterministic, "outliers": flags}), flush=True)
    return 0 if deterministic else 3


if __name__ == "__main__":
    sys.exit(main())
