"""FASE 6: inferencia y CLI `infer` (architecture.md §8)."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.predictor import Predictor, read_images
from src.main import main
from src.model.cnn import CNN
from src.training.trainer import Trainer

SHARED = Path(__file__).resolve().parents[3] / "shared"
INITIAL = SHARED / "initial_weights" / "initial_weights.tb"
pytestmark = pytest.mark.skipif(not (SHARED / "dataset" / "test_images.bin").exists(), reason="dataset no generado")


@pytest.fixture(scope="module")
def images():
    return np.fromfile(SHARED / "dataset" / "test_images.bin", dtype=np.uint8).reshape(-1, 3072)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    Trainer(SHARED, out, "t", epochs=1, log=lambda *_: None).train()
    return out / "model_final.tb"


def test_logits_match_training_evaluation(images, trained):
    """El predictor reproduce, bit a bit, los logits que el trainer escribió en test_logits.csv."""
    with open(trained.parent / "test_logits.csv") as f:
        ref = np.array([float(r["logit"]) for r in csv.DictReader(f)], dtype=np.float32)
    assert np.array_equal(Predictor.load(trained).logits(images), ref)


def test_output_fields_and_consistency(images, trained):
    pr = Predictor.load(trained)
    z = pr.logits(images[:50])
    for zi, r in zip(z, pr.predict_batch(images[:50])):
        assert set(r) == {"prediction", "probability", "confidence"}
        assert r["prediction"] == ("DOG" if zi >= 0 else "NOT_DOG")
        assert 0 <= r["probability"] <= 1 and 0.5 <= r["confidence"] <= 1
        assert abs(r["confidence"] - max(r["probability"], 1 - r["probability"])) <= 1e-9


def test_single_equals_batch_and_chunking(images, trained):
    pr = Predictor.load(trained)
    batch = pr.predict_batch(images[:70])                    # cruza el límite de 32
    assert [pr.predict(images[i]) for i in (0, 33, 69)] == [batch[0], batch[33], batch[69]]


class _FakeModel:
    """Devuelve logits fijados: aísla la conversión logit → JSON del resto de la red."""
    def __init__(self, z):
        self.z = np.array(z, dtype=np.float32)

    def forward(self, x):
        return self.z[: len(x)].reshape(-1, 1)


def test_logit_to_json_hand_computed():
    z = [0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, -0.0]
    r = Predictor(_FakeModel(z)).predict_batch(np.zeros((len(z), 3072), dtype=np.uint8))
    assert r[0] == {"prediction": "DOG", "probability": 0.5, "confidence": 0.5}          # z = 0 ⇒ DOG
    assert r[1] == {"prediction": "DOG", "probability": 0.731, "confidence": 0.731}
    assert r[2] == {"prediction": "NOT_DOG", "probability": 0.269, "confidence": 0.731}
    assert r[3] == {"prediction": "DOG", "probability": 1.0, "confidence": 1.0}
    assert r[4] == {"prediction": "NOT_DOG", "probability": 0.0, "confidence": 1.0}
    assert r[5]["probability"] == 1.0 and r[6] == {"prediction": "NOT_DOG", "probability": 0.0, "confidence": 1.0}
    assert r[7]["prediction"] == "DOG"                                                   # -0.0 ≥ 0


def test_cli_image_and_batch(images, trained, tmp_path, capsys):
    img, bat = tmp_path / "i.bin", tmp_path / "b.bin"
    images[5].tofile(img); images[:3].tofile(bat)
    assert main(["infer", "--model", str(trained), "--image", str(img)]) == 0
    one = json.loads(capsys.readouterr().out)
    assert isinstance(one, dict)
    assert main(["infer", "--model", str(trained), "--batch", str(bat)]) == 0
    many = json.loads(capsys.readouterr().out)
    assert isinstance(many, list) and len(many) == 3
    assert many == Predictor.load(trained).predict_batch(images[:3])


def test_cli_rejects_bad_input(images, trained, tmp_path, capsys):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 100)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    two = tmp_path / "two.bin"
    images[:2].tofile(two)
    m = str(trained)
    assert main(["infer", "--model", m, "--image", str(bad)]) == 2
    assert main(["infer", "--model", m, "--batch", str(bad)]) == 2
    assert main(["infer", "--model", m, "--batch", str(empty)]) == 2
    assert main(["infer", "--model", m, "--image", str(two)]) == 2           # 2 imágenes con --image
    assert main(["infer", "--model", m]) == 2
    assert main(["infer", "--model", m, "--image", str(bad), "--batch", str(bad)]) == 2
    assert main(["infer", "--model", str(tmp_path / "no.tb"), "--image", str(two)]) == 2
    corrupt = tmp_path / "c.tb"
    corrupt.write_bytes(b"basura")
    assert main(["infer", "--model", str(corrupt), "--batch", str(two)]) == 2


def test_initial_weights_model_also_loads(images):
    assert len(Predictor(CNN.load(INITIAL)).predict_batch(images[:2])) == 2
