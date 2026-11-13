"""Exporta imágenes del conjunto de test como PNG para probar el motor a mano. Pillow (Python del sistema).

  python3 scripts/exportar_muestras.py [--n 10] [--out muestras]
Nombre de salida: muestras/test_0012_perro.png (la etiqueta real va en el nombre; el motor no la ve).
Tamaño original 32×32 (los PNG de ese tamaño se convierten sin pérdida) y una copia ampliada ×8 para verlas.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="cuántas de cada clase")
    ap.add_argument("--out", default="muestras")
    a = ap.parse_args()
    d = REPO / "shared" / "dataset"
    img = np.fromfile(d / "test_images.bin", dtype=np.uint8).reshape(-1, 3, 32, 32)
    lab = np.fromfile(d / "test_labels.bin", dtype=np.uint8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for clase, nombre in ((1, "perro"), (0, "no_perro")):
        for pos in np.flatnonzero(lab == clase)[:a.n]:
            im = Image.fromarray(img[pos].transpose(1, 2, 0), "RGB")
            im.save(out / f"test_{pos:04d}_{nombre}.png")
            im.resize((256, 256), Image.NEAREST).save(out / f"test_{pos:04d}_{nombre}_grande.png")
    print(f"{2 * a.n} imágenes en {out}/ (cada una con su copia _grande.png para verla)")


if __name__ == "__main__":
    main()
