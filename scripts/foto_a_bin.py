"""Convierte fotos reales (JPG, PNG, WebP…) al formato crudo del proyecto: 3072 bytes uint8, planar R/G/B, 32×32.

Los motores NO decodifican imágenes (architecture.md §8); este script es una herramienta aparte. Requiere Pillow:
usa el Python del sistema (`python3`), no el entorno virtual del motor.

  python3 scripts/foto_a_bin.py foto.jpg                      # escribe foto.bin
  python3 scripts/foto_a_bin.py a.jpg b.png --batch lote.bin  # varias fotos en un solo fichero (--batch)
  python3 scripts/foto_a_bin.py foto.jpg --clasificar         # además llama al motor y muestra el resultado
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIDE = 32
REPO = Path(__file__).resolve().parents[1]


def a_bytes(path):
    """Foto → 3072 bytes. Respeta la orientación EXIF, aplana la transparencia sobre blanco, recorta al cuadrado
    centrado y reduce a 32×32 (LANCZOS). Orden de bytes: todo el canal R, luego G, luego B (como CIFAR-10)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            fondo = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(fondo, im)
        im = im.convert("RGB")
        w, h = im.size
        lado = min(w, h)
        im = im.crop(((w - lado) // 2, (h - lado) // 2, (w - lado) // 2 + lado, (h - lado) // 2 + lado))
        if lado != SIDE:
            im = im.resize((SIDE, SIDE), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)                 # (32, 32, 3) HWC
    return np.ascontiguousarray(arr.transpose(2, 0, 1)).reshape(-1)   # CHW plano


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fotos", nargs="+")
    ap.add_argument("--batch", help="escribe todas las fotos en este único fichero (para `infer --batch`)")
    ap.add_argument("--clasificar", action="store_true", help="llama a `main.py infer` del motor Python con el resultado")
    ap.add_argument("--modelo", default=str(REPO / "implementations/python/runs/reference_001/model_final.tb"))
    a = ap.parse_args(argv)
    datos = []
    for f in a.fotos:
        try:
            datos.append(a_bytes(f))
        except (OSError, ValueError) as e:
            print(f"error con {f}: {e}", file=sys.stderr)
            return 2
    if a.batch:
        Path(a.batch).write_bytes(np.concatenate(datos).tobytes())
        salidas = [a.batch]
    else:
        salidas = []
        for f, d in zip(a.fotos, datos):
            out = str(Path(f).with_suffix(".bin"))
            Path(out).write_bytes(d.tobytes())
            salidas.append(out)
    for s in salidas:
        print(f"escrito {s}")
    if a.clasificar:
        motor = REPO / "implementations/python"
        for s, nombre in ([(salidas[0], "lote")] if a.batch else zip(salidas, a.fotos)):
            flag = "--batch" if a.batch else "--image"
            r = subprocess.run([str(motor / ".venv/bin/python"), "-m", "src.main", "infer", "--model", str(Path(a.modelo).resolve()),
                                flag, str(Path(s).resolve())],       # el motor se ejecuta desde su carpeta: rutas absolutas
                               cwd=motor, capture_output=True, text=True)
            if r.returncode:
                print(r.stderr, file=sys.stderr)
                return r.returncode
            print(f"{nombre}: {r.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
