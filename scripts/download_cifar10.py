"""Descarga CIFAR-10 (versión binaria) a shared/dataset/raw/ y verifica su integridad (docs/dataset.md §2).

Uso: implementations/python/.venv/bin/python -I scripts/download_cifar10.py
- Comprueba el MD5 oficial publicado en la página de la fuente.
- Calcula el SHA-256 y lo registra en shared/dataset/CHECKSUMS.sha256 (primera descarga: se fija, no se inventa).
- Si el fichero ya existe y su MD5 es correcto, no vuelve a descargar.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5_OFICIAL = "c32a1d4ab5d03f1284b67883e8d87530"
NAME = "cifar-10-binary.tar.gz"
RAW = ROOT / "shared" / "dataset" / "raw"
CHECKSUMS = ROOT / "shared" / "dataset" / "CHECKSUMS.sha256"


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / NAME
    if not (dest.exists() and digests(dest)[0] == MD5_OFICIAL):
        tmp = dest.with_suffix(".part")
        print(f"descargando {URL}")
        with urllib.request.urlopen(URL, timeout=60) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.replace(dest)
    md5, sha = digests(dest)
    if md5 != MD5_OFICIAL:
        dest.unlink()
        sys.exit(f"MD5 incorrecto: {md5} (oficial {MD5_OFICIAL}); fichero eliminado")
    print(f"MD5 verificado: {md5}\nSHA-256: {sha}\ntamaño: {dest.stat().st_size} B")
    lines = [l for l in CHECKSUMS.read_text().splitlines() if not l.endswith(f"  raw/{NAME}")] if CHECKSUMS.exists() else []
    CHECKSUMS.write_text("\n".join(lines + [f"{sha}  raw/{NAME}"]) + "\n")


if __name__ == "__main__":
    main()
