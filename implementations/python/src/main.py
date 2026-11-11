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
                            str(ROOT / "tests" / "test_training.py")]))


def cmd_train(args):
    from .training.trainer import Trainer
    out = Path(args.out) / args.run
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} ya existe y no está vacío (no se sobrescriben resultados)", file=sys.stderr)
        return 2
    m = Trainer(args.data_dir, out, args.run, args.epochs).train()
    print(m)
    return 0


def cmd_infer(args):
    print("infer: pendiente (FASE 6)", file=sys.stderr)
    return 2


def main(argv=None):
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tests"); t.add_argument("--data-dir", default=str(DEFAULT_DATA)); t.set_defaults(f=cmd_tests)
    tr = sub.add_parser("train"); tr.add_argument("--data-dir", default=str(DEFAULT_DATA))
    tr.add_argument("--out", default=str(ROOT / "runs")); tr.add_argument("--run", default="reference_001")
    tr.add_argument("--epochs", type=int, default=20); tr.set_defaults(f=cmd_train)
    i = sub.add_parser("infer"); i.add_argument("--model"); i.add_argument("--image"); i.add_argument("--batch")
    i.set_defaults(f=cmd_infer)
    args = p.parse_args(argv)
    return args.f(args)


if __name__ == "__main__":
    sys.exit(main())
