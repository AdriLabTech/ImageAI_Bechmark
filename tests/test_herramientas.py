"""Herramientas de usuario (scripts/foto_a_bin.py, exportar_muestras.py) y utilidades estadísticas del arnés."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import foto_a_bin  # noqa: E402

HAY_DATOS = (REPO / "shared/dataset/test_images.bin").exists()


@pytest.mark.skipif(not HAY_DATOS, reason="dataset no generado")
def test_png_32px_roundtrip_is_lossless(tmp_path):
    """Una imagen del dataset → PNG → foto_a_bin da exactamente los 3072 bytes originales."""
    img = np.fromfile(REPO / "shared/dataset/test_images.bin", dtype=np.uint8).reshape(-1, 3072)
    for k in (0, 7, 1999):
        p = tmp_path / f"{k}.png"
        Image.fromarray(img[k].reshape(3, 32, 32).transpose(1, 2, 0), "RGB").save(p)
        assert np.array_equal(foto_a_bin.a_bytes(p), img[k])


def test_channel_order_is_planar_rgb(tmp_path):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 10, 20, 30
    Image.fromarray(arr, "RGB").save(tmp_path / "c.png")
    b = foto_a_bin.a_bytes(tmp_path / "c.png")
    assert b.size == 3072 and (b[:1024] == 10).all() and (b[1024:2048] == 20).all() and (b[2048:] == 30).all()


def test_large_nonsquare_photo_is_cropped_and_resized(tmp_path):
    Image.new("RGB", (640, 480), (200, 50, 50)).save(tmp_path / "g.jpg")
    b = foto_a_bin.a_bytes(tmp_path / "g.jpg")
    assert b.size == 3072 and abs(int(b[:1024].mean()) - 200) <= 3


def test_transparency_is_flattened_on_white(tmp_path):
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(tmp_path / "t.png")
    assert (foto_a_bin.a_bytes(tmp_path / "t.png") == 255).all()


def test_cli_rejects_non_image_and_writes_batch(tmp_path):
    bad = tmp_path / "no.jpg"
    bad.write_bytes(b"esto no es una imagen")
    assert foto_a_bin.main([str(bad)]) == 2
    Image.new("RGB", (32, 32), (1, 2, 3)).save(tmp_path / "a.png")
    Image.new("RGB", (50, 40), (1, 2, 3)).save(tmp_path / "b.png")
    out = tmp_path / "lote.bin"
    assert foto_a_bin.main([str(tmp_path / "a.png"), str(tmp_path / "b.png"), "--batch", str(out)]) == 0
    assert out.stat().st_size == 2 * 3072


@pytest.mark.skipif(not (REPO / "implementations/python/runs/reference_001/model_final.tb").exists(), reason="sin modelo entrenado")
def test_clasificar_with_relative_paths(tmp_path, monkeypatch, capsys):
    """Regresión: --clasificar fallaba con rutas relativas porque el motor corre desde su propia carpeta."""
    Image.new("RGB", (40, 40), (120, 80, 40)).save(tmp_path / "f.png")
    monkeypatch.chdir(tmp_path)
    assert foto_a_bin.main(["f.png", "--clasificar"]) == 0
    assert '"prediction"' in capsys.readouterr().out
