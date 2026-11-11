# Motor Python (ImageAI-Benchmark)

Motor #1: Python 3 + NumPy, sin frameworks de ML. Contrato en `../../docs/` (`model.md`, `python.md`).

**Estado (FASE 6)**: operaciones, capas, `CNN`, SGD, cargador de datos, entrenamiento, checkpoints, inferencia y CLI (`tests`, `train`, `infer`) implementados y probados.

Ejecutar: `.venv/bin/python -m src.main train --run reference_001` (resultados en `runs/`, ignorado por git) `.venv/bin/python -m src.main tests` y `.venv/bin/python -m src.main infer --model runs/reference_001/model_final.tb --image img.bin` (o `--batch`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                      # 118 pruebas: operaciones, modelo, gradcheck FP64 (capas y red completa), T1/T2 contra shared/test_vectors
.venv/bin/pytest -q ../../tests          # PRNG y regeneración bit a bit de artefactos
../../tests/check_spec.py                # aritmética de la especificación (python3 -I)
```

Regenerar artefactos compartidos: `.venv/bin/python ../../scripts/generate_artifacts.py`.
Versiones usadas: Python 3.14.10, NumPy 2.5.4, pytest 9.2.0.
El motor no importa `numpy.random` ni nada de `scripts/`.
