# Motor Python (ImageAI-Benchmark)

Motor #1: Python 3 + NumPy, sin frameworks de ML. Contrato en `../../docs/` (`model.md`, `python.md`).

**Estado (FASE 3)**: operaciones, capas, `CNN` y SGD implementados y probados; datos (FASE 4), entrenamiento (FASE 5) e inferencia (FASE 6) pendientes.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                      # 108 pruebas: operaciones, modelo, gradcheck FP64 (capas y red completa), T1/T2 contra shared/test_vectors
.venv/bin/pytest -q ../../tests          # PRNG y regeneración bit a bit de artefactos
../../tests/check_spec.py                # aritmética de la especificación (python3 -I)
```

Regenerar artefactos compartidos: `.venv/bin/python ../../scripts/generate_artifacts.py`.
Versiones usadas: Python 3.14.10, NumPy 2.5.4, pytest 9.2.0.
El motor no importa `numpy.random` ni nada de `scripts/`.
