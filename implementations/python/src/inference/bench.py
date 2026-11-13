"""Mediciones de inferencia del benchmark (benchmark.md §5.2-§5.3). Solo cronometra; no calcula métricas de calidad."""
import time

import numpy as np

from ..data.dataloader import Dataset
from ..data.preprocessing import to_float32
from .predictor import Predictor

LAT_WARMUP, LAT_N = 100, 1000
THR_WARMUP, THR_N, THR_BATCH = 100, 100, 32


def bench_inference(model_path, data_dir):
    """Latencia (lote 1, normalización incluida, modelo ya cargado) y throughput (lote de 32 ya normalizado, solo forward)."""
    t0 = time.perf_counter()
    pr = Predictor.load(model_path)
    load_s = time.perf_counter() - t0
    images = Dataset(data_dir, "test").images
    lat = []
    for i in range(LAT_WARMUP + LAT_N):
        img = images[i % len(images)]
        t = time.perf_counter_ns()
        pr.predict(img)
        dt = (time.perf_counter_ns() - t) / 1e9
        if i >= LAT_WARMUP:
            lat.append(dt)
    x = to_float32(images[:THR_BATCH])
    thr = []
    for i in range(THR_WARMUP + THR_N):
        t = time.perf_counter_ns()
        pr.model.forward(x)
        dt = (time.perf_counter_ns() - t) / 1e9
        if i >= THR_WARMUP:
            thr.append(THR_BATCH / dt)
    return {"model_load_s": round(load_s, 6), "latency_s": [round(v, 6) for v in lat],
            "throughput_img_s": [round(v, 3) for v in thr]}
