"""CLI del motor Python (docs/architecture.md §8): tests | train | infer."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # mono-hilo (D-002), antes de importar NumPy
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT.parents[1] / "shared"


def cmd_tests(args):
    import pytest
    return int(pytest.main(["-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_golden_vectors.py"),
                            str(ROOT / "tests" / "test_training.py"),
                            str(ROOT / "tests" / "test_inference.py")]))


def cmd_train(args):
    import json
    from .training.trainer import Trainer
    out = Path(args.out) / args.run
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} ya existe y no está vacío (no se sobrescriben resultados)", file=sys.stderr)
        return 2
    tr = Trainer(args.data_dir, out, args.run, args.epochs, warmup=args.warmup)
    m = tr.train()
    if args.perf_out:
        Path(args.perf_out).write_text(json.dumps(tr.perf) + "\n", encoding="utf-8")
    print(m)
    return 0


def cmd_bench_infer(args):
    import json

    from .inference.bench import bench_inference
    Path(args.out).write_text(json.dumps(bench_inference(args.model, args.data_dir)) + "\n", encoding="utf-8")
    return 0


def cmd_infer(args):
    import json

    from .inference.predictor import Predictor, read_images
    if bool(args.image) == bool(args.batch):
        print("error: indica exactamente uno de --image o --batch", file=sys.stderr)
        return 2
    try:
        images = read_images(args.image or args.batch, single=bool(args.image))
        res = Predictor.load(args.model).predict_batch(images)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res[0] if args.image else res, allow_nan=False))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tests"); t.add_argument("--data-dir", default=str(DEFAULT_DATA)); t.set_defaults(f=cmd_tests)
    tr = sub.add_parser("train"); tr.add_argument("--data-dir", default=str(DEFAULT_DATA))
    tr.add_argument("--out", default=str(ROOT / "runs")); tr.add_argument("--run", default="reference_001")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--warmup", action="store_true", help="benchmark: 1 época de calentamiento sobre una copia")
    tr.add_argument("--perf-out", help="benchmark: JSON con los tiempos del motor"); tr.set_defaults(f=cmd_train)
    bi = sub.add_parser("bench-infer"); bi.add_argument("--model", required=True); bi.add_argument("--out", required=True)
    bi.add_argument("--data-dir", default=str(DEFAULT_DATA)); bi.set_defaults(f=cmd_bench_infer)
    i = sub.add_parser("infer"); i.add_argument("--model"); i.add_argument("--image"); i.add_argument("--batch")
    i.set_defaults(f=cmd_infer)
    args = p.parse_args(argv)
    return args.f(args)


if __name__ == "__main__":
    sys.exit(main())
