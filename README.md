# ImageAI-Benchmark

Clasificador neuronal de imágenes (**¿perro o no perro?**) implementado desde cero en **seis lenguajes** para comparar capacidades, rendimiento y complejidad de cada uno al construir motores de IA.

**Estado**: FASE 2 hecha (operaciones y capas del motor Python con 69 pruebas, 2026-11-11). Siguiente: FASE 3 (integración del modelo).

## Filosofía

1. **Aprendizaje real**: CNN propia (Conv2D → ReLU → MaxPool ×2 → Dense → logit), sin modelos preentrenados ni APIs externas.
2. **Implementación desde cero**: convoluciones, activaciones, backpropagation, pérdida y entrenamiento manuales. NumPy para matemática de tensores; **prohibidos** PyTorch/TensorFlow/Keras/JAX/autograd.
3. **Independencia entre lenguajes**: cada motor funciona solo; el intercambio ocurre vía artefactos en disco (`shared/`), nunca entre programas.
4. **Equivalencia matemática**: los seis ejecutan exactamente la misma arquitectura y el mismo entrenamiento (contrato v1).
5. **Transparencia experimental**: toda decisión, optimización y limitación se documenta; nada se ajusta para favorecer resultados.

## Estructura

```
docs/              Contratos: model · dataset · training · benchmark · architecture · python · decisions
implementations/   Motores medibles (python primero; c, rust, cpp, java, go después)
shared/            Artefactos canónicos: dataset · initial_weights · permutations · test_vectors
scripts/           Herramientas compartidas (solo Python, no se miden)
benchmark/         Arnés de medición + resultados crudos
reports/           Informes: línea base de referencia y comparativas
tests/             Pruebas transversales del repositorio
```

## Documentación contractual

| Documento | Contenido |
|---|---|
| [docs/model.md](docs/model.md) | Arquitectura CNN-32, fórmulas forward/backward, inicialización He/Xavier, BCE con logits, reglas FP32 |
| [docs/dataset.md](docs/dataset.md) | CIFAR-10, subconjunto binario 8000/1000/2000, selección estratificada determinista, formatos binarios |
| [docs/training.md](docs/training.md) | Ciclo exacto: 20 épocas, lotes de 32, SGD lr 0.01, checkpoints, logs, control de divergencia |
| [docs/benchmark.md](docs/benchmark.md) | Cinco categorías (rendimiento, recursos, calidad, complejidad, portabilidad), procedimiento y formatos |
| [docs/architecture.md](docs/architecture.md) | Contratos entre los seis motores: TensorBundle v1, CLI, vectores de prueba, tolerancias |
| [docs/python.md](docs/python.md) | Propuesta del motor Python: módulos, estrategia im2col+GEMM, pruebas |
| [docs/decisions.md](docs/decisions.md) | Decisiones D-001…D-033 (sin pendientes) |

## Resumen técnico

- **Modelo**: CNN-32, 78.945 parámetros, entrada RGB 32×32 NCHW FP32 [0,1], salida logit → sigmoid.
- **Datos**: CIFAR-10 → binario DOG/NOT_DOG, 8.000/1.000/2.000 equilibrados, manifiestos deterministas (splitmix64, semilla 20261110).
- **Entrenamiento**: 20 épocas × 250 lotes de 32, SGD puro lr 0,01, BCE con logits estable, FP32, sin azar en runtime, modelo final = época 20.
- **Benchmark**: CPU mono-hilo, 5 repeticiones + calentamiento, mediana + p95, arnés externo idéntico para los seis, corrección matemática separada del rendimiento.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Especificación contractual | Terminada |
| 1 | Validación documental cruzada | **Hecha** (`docs/validation-fase1.md`) |
| 2 | Motor matemático Python + vectores dorados | **Hecha** (18 vectores, pesos iniciales, gradcheck) |
| 3 | Integración del modelo | Pendiente |
| 4 | Dataset | Pendiente |
| 5 | Entrenamiento | Pendiente |
| 6 | Inferencia | Pendiente |
| 7 | Benchmark inicial Python | Pendiente |
| 8 | Documentación final | Pendiente |

## Uso

*Aún no hay ejecutable. Cuando exista (FASE 2+), el contrato de comandos es:*

```
python -m src.main tests  --data-dir ../shared/
python -m src.main train  --data-dir ../shared/ --out runs/
python -m src.main infer  --model runs/model_final.tb --image imagen.bin
```
