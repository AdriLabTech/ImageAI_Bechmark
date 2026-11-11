# python.md — Propuesta de arquitectura del motor Python

> **Estado**: FASE 0 (propuesta). Versión 1.0-draft. Pendiente de aprobación.
> Define la organización del primer motor, `implementations/python/`. No hay código todavía: esto es el contrato interno del motor, alineado con `docs/architecture.md` §4 (interfaces concepto) y §8 (CLI).

---

## 1. Objetivo y restricciones

Motor de referencia y primer competidor del benchmark. Restricciones:

- **Permitido**: Python 3 + NumPy para matemática de tensores (matmul, broadcasting, reducciones).
- **Prohibido**: PyTorch, TensorFlow, Keras, JAX, scikit-learn, autograd, o cualquier biblioteca que implemente capas, entrenamiento o autodiferenciación.
- **Pruebas**: pytest.
- **Matplotlib**: solo en herramientas de análisis fuera del motor medido (no forma parte del binario/código cronometrado).
- **Sin azar en runtime**: el motor no importa `numpy.random` ni `random`. Todos los artefactos aleatorios provienen de `shared/`.
- **FP32 estricto**: `np.float32` en todo el grafo; sin conversiones a float64 en entrenamiento (las pruebas de gradientes usan float64 internamente como herramienta, no el motor).
- El motor **no descarga datos, no lee la fuente original, no importa de `scripts/`**: consume únicamente `shared/` (independencia, `docs/architecture.md` §3).

## 2. Organización de módulos

Basada en la propuesta original, con un ajuste justificado: `cifar10.py` se muda a `scripts/prepare_dataset.py`. Razón: leer la fuente original es una tarea de la **herramienta compartida**, no del motor (decisión D-011): ningún motor leerá la fuente original, y el motor Python no puede ser una excepción o la comparación de dependencias y caminos de datos quedaría falseada.

```text
implementations/python/
│
├── src/
│   ├── core/
│   │   ├── tensor_utils.py     # I/O TensorBundle v1, validación de formas, comparación con tolerancias
│   │   ├── initializers.py     # carga de pesos compartidos (no genera aleatoriedad)
│   │   ├── activations.py      # relu, sigmoid (forward + derivadas)
│   │   └── losses.py           # BCE con logits (estable) + gradiente
│   │
│   ├── layers/
│   │   ├── conv2d.py           # Conv2D stride 1 VALID, im2col + matmul (ver §4)
│   │   ├── maxpool2d.py        # 2×2 stride 2 floor, argmax primer-orden-raster
│   │   ├── flatten.py          # reshape NCHW
│   │   ├── relu.py             # envoltorio de activations.relu_* como capa
│   │   └── dense.py            # capa densa (O, I)
│   │
│   ├── model/
│   │   └── cnn.py              # CNN-32: composición de capas, forward/backward completos
│   │
│   ├── training/
│   │   ├── optimizer.py        # SGD puro (lr 0.01, sin estado)
│   │   ├── trainer.py          # ciclo de training.md §3, validación, logs
│   │   └── checkpoint.py       # escritura de checkpoints TensorBundle + meta
│   │
│   ├── data/
│   │   ├── preprocessing.py    # uint8 → float32 /255.0 (bit-exacto)
│   │   └── dataloader.py       # lectura COMPLETA a RAM de .bin/.perm (D-032); composición de lotes
│   │
│   ├── inference/
│   │   └── predictor.py       # carga de modelo, predict / predict_batch → JSON
│   │
│   └── main.py                # CLI: tests | train | infer (architecture.md §8)
│
├── tests/
│   ├── gradcheck.py            # utilidad de diferencias finitas FP64 (solo pruebas)
│   ├── test_tensor_utils.py    # ida/vuelta TensorBundle, validaciones
│   ├── test_conv2d.py          # formas, casos 0/delta, gradcheck
│   ├── test_maxpool2d.py       # par/impar, empates, gradiente
│   ├── test_dense.py
│   ├── test_activations.py
│   ├── test_losses.py          # estabilidad numérica BCE
│   ├── test_gradients.py       # gradcheck FP64 de TODA la red por parámetro
│   ├── test_model.py           # dimensiones capa a capa, forward/backward completos
│   └── test_golden_vectors.py # T1 + T2 contra shared/test_vectors/
│
├── requirements.txt            # numpy, pytest (matplotlib en análisis, aparte)
└── README.md                   # cómo ejecutar tests/train/infer, versiones usadas
```

## 3. Contratos de los módulos

| Módulo | Responsabilidad | Interfaz clave |
|---|---|---|
| `core/tensor_utils.py` | Formato TensorBundle v1 (lectura/escritura exacta de `architecture.md` §5), comparación con tolerancias | `read_tb(ruta) → dict[nombre, np.ndarray]`, `write_tb(ruta, dict)`, `assert_close(a, b, atol, rtol)` |
| `core/initializers.py` | Cargar `initial_weights.tb` y validar formas esperadas de las 8 capas | `load_initial_weights(ruta)` |
| `core/activations.py` | ReLU y Sigmoid con sus derivadas, punto x=0 definido | `relu_forward`, `relu_backward`, `sigmoid_forward` (estable), `sigmoid_grad` |
| `core/losses.py` | `bce_with_logits(z, y) → (pérdida, dZ)` según `model.md` §8 | `bce_with_logits` |
| `layers/*` | Una clase por capa con `forward`, `backward`, `params`, `grads`, `zero_grads`, validación de dimensiones | contrato §4.1 de `architecture.md` |
| `model/cnn.py` | `CNN` con las 10 capas en orden, nombres canónicos, `save`/`load` | contrato §4.2 |
| `training/optimizer.py` | `SGD(lr=0.01).step(params, grads)` puro | — |
| `training/trainer.py` | Ejecuta `training.md` §3 exacto; escribe `train_log.jsonl`, `final_metrics.json`, `test_logits.csv` | `Trainer.train()` |
| `training/checkpoint.py` | `checkpoint_epoch_NNN.tb` + `meta.*` escalares | — |
| `data/preprocessing.py` | `to_float32(u8) → np.float32 /255.0` | — |
| `data/dataloader.py` | `Dataset` con `{split}_images.bin`/`_labels.bin` leídos enteros a RAM antes de cronometrar (`np.fromfile`, sin memmap); `batch(posiciones)`; `EpochPermutations(e)` | — |
| `inference/predictor.py` | `Predictor.load(modelo)`, `predict(img3072)` → dict JSON | contrato §8 |

## 4. Estrategia interna documentada (D-001 aplicado a Python)

- **Conv2D**: **im2col + `np.matmul`** (BLAS). El motor documenta esta estrategia como su elección. Forward: despliegue de parches (N·C·K·K, H_out·W_out) → GEMM contra pesos aplanados. Backward: `dW` como GEMM sobre im2col, `dX` por *col2im* (acumulación dispersa).
- **MaxPool2D**: reshape a `(N, C, H//2, 2, W//2, 2)` y `np.max`/`np.argmax` sobre los ejes de ventana; `argmax` de NumPy devuelve el **primer máximo en orden C**, que coincide con la regla de empates del contrato (`model.md` §4.3) **siempre que** los ejes de ventana queden contiguos y en orden (r, s): hay que recortar a `[:12, :12]` con entrada 13×13 y transponer a `(N, C, H//2, W//2, 2, 2)` antes de aplanar la ventana a 4.
- **Dense**: `Y = X @ W.T + B` (W en forma canónica `(O, I)`).
- **Sumas/reducciones**: `np.sum`/`np.mean` (suma por pares de NumPy, determinista en la misma versión).
- Alternativa descartada: bucles directos por elemento (prohibitivamente lento en Python; la decisión de contrato permite estrategia libre por motor, D-001).

Las reglas de precisión de `model.md` §9 se aplican: FP32 en todo el grafo; el motor registra en `summary.json.meta` la versión de Python, NumPy y el BLAS detectado (`np.show_config()`), porque el backend BLAS afecta al rendimiento y debe quedar documentado, no oculto.

## 5. Pruebas

1. **Unitarias por operación** (`test_conv2d.py`, …): formas, casos triviales (kernel delta, bias cero, entrada constante), comportamiento en bordes (pool impar, empates, logits extremos).
2. **Gradient check** (`test_gradients.py` + `gradcheck.py`): diferencias centrales en float64 con `h = 1e-5`, criterios de `model.md` §10 (rel 1e-6 dense/BCE, 1e-5 conv/pool). Se ejecuta sobre **cada parámetro** del modelo completo y sobre entradas.
3. **Vectores dorados** (`test_golden_vectors.py`): T1 por operación y T2 del paso de entrenamiento contra `shared/test_vectors/`, con reporte de violación máxima.
4. **Determinismo** (parte de `main.py tests`): dos entrenamientos cortos (1 época, semilla-artefacto idéntica) → SHA-256 de checkpoint idéntico.
5. **I/O**: ida y vuelta de TensorBundle con casos con nombre, dtype y ndim variados.

Orden de desarrollo (FASE 2): `tensor_utils` → `initializers`/`activations`/`losses` (con gradcheck) → `dense`/`flatten` → `maxpool2d` → `conv2d` → backprop completa (`cnn`) — cada uno con sus pruebas antes de integrarse.

## 6. Salidas del motor

```text
implementations/python/runs/<id_ejecucion>/
├── checkpoint_epoch_001.tb … checkpoint_epoch_020.tb
├── checkpoint_latest.tb
├── model_final.tb
├── train_log.jsonl
├── final_metrics.json
└── test_logits.csv
```

El CLI reproduce exactamente el contrato de `architecture.md` §8 (`tests`/`train`/`infer` con los mismos flags).

## 7. Versiones y entorno

- Python 3.14 y NumPy 2.5 disponibles en la máquina de desarrollo (registradas en `meta` de cada resultado).
- `requirements.txt` minimalista: `numpy`, `pytest`. Matplotlib queda fuera del motor (solo análisis puntual posterior).
- Entorno virtual propio del motor (`.venv` gitignored) para aislar versiones del resto de herramientas.

## 8. Criterios de aceptación de FASE 2 (motor matemático)

1. `pytest` en verde: operaciones + gradientes + golden vectors.
2. Gradcheck FP64 dentro de tolerancias para los 8 bloques de parámetros y las entradas.
3. T1 completo (`test_golden_vectors.py`) contra los vectores generados.
4. La serialización TensorBundle ida/vuelta es byte-idéntica (reescritura produce el mismo SHA-256).

## 9. Estado (2026-11-11, tras la FASE 3)

**Implementado y probado** (`implementations/python/`, 108 pruebas del motor + 4 transversales):
`tensor_utils` (TensorBundle), `initializers` (carga), `activations`, `losses`, `Dense`, `Flatten`, `ReLU`, `MaxPool2D`, `Conv2D` (im2col + GEMM), `model/cnn.py` (`CNN`: 10 capas, `forward`/`backward`, `params`/`grads`, `save`/`load`), `training/optimizer.py` (SGD), `tests/gradcheck.py`.
**Pendiente**: `training/trainer.py` y `checkpoint.py` (FASE 5), `data/*` (FASE 4), `inference/*` (FASE 6), `main.py`, determinismo de entrenamiento completo, caso `train_step/case_002` con lote real.

## 10. Resultados medidos (FASES 2 y 3)

- **Gradcheck de capas** (FP64, h = 1e-5, muestra de 400 elementos por tensor): dense/BCE/ReLU/sigmoid < 1e-6; conv/maxpool < 1e-5, con las formas reales de conv1, conv2, dense1 y dense2.
- **Gradcheck del modelo completo + BCE** (5 semillas, 150 elementos por tensor, respecto a los 8 tensores de parámetros y a la entrada): errores relativos ≈ 1e-10…1e-8. Un tensor de una semilla necesitó repetir con h = 1e-7 por un cruce de argmax de MaxPool (verificado: con h = 1e-7 coincide con el analítico hasta 6e-10 absoluto).
- **T1** (20 vectores, incluido `model`) y **T2** (`train_step/case_001`, lote sintético): error absoluto máximo ≤ 1,1e-5 (`conv1.grad_weight`); error escalado por tensor ≤ 1,1e-6; el resto de comparaciones ≤ 1e-6.
- **Consistencia**: el gradiente de un lote de 8 es la media de los 8 gradientes individuales (coherencia `dZ = (σ−y)/N`); `forward` por lote y por muestra coinciden a 1e-5; `save`→`load` byte-idéntico.
- **Descenso**: 60 pasos de SGD (lr 0,01) sobre 16 imágenes fijas bajan la pérdida de 0,9381 a 0,4047. Es una prueba de sanidad del sentido del gradiente, no una métrica de calidad.
- **Mutaciones** (fallos introducidos a propósito, todos detectados): ReLU en 0, empate de MaxPool al último, `col2im` sin acumular, BCE sin estabilizar, `dense1.dx ×1,001`, gradiente de bias de conv ×0,99, backward que salta una ReLU, SGD con signo opuesto. Un `lr` erróneo del 1 % o 0,1 % **solo** lo detecta el criterio escalado (P-016).
- Referencia indicativa (no es benchmark; 1 muestra, BLAS sin fijar a un hilo): un paso de entrenamiento de 32 imágenes ≈ 24 ms.

## 11. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción (FASE 0). |
| 1.0-draft r1 | 2026-11-11 | FASE 1: precisión sobre el aplanado de ventana de MaxPool (ver `docs/validation-fase1.md`). |
| 1.0-draft r2 | 2026-11-11 | Sin memmap (D-032): datos en RAM antes del cronómetro. |
| 1.0-draft r3 | 2026-11-11 | FASE 2: estado, `relu.py`, resultados medidos. |
| 1.0-draft r4 | 2026-11-11 | FASE 3: `CNN`, SGD, gradcheck del modelo completo, resultados medidos. |
