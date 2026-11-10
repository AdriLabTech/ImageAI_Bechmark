# benchmark.md — Especificación contractual del procedimiento de medición

> **Estado**: FASE 0 (especificación). Versión 1.0-draft. Pendiente de aprobación.
> Documento **contractual**. Define cómo se mide, con qué condiciones y con qué formato se reporta. Regla de transparencia: **todo número publicado debe trazarse a un fichero de resultados producido por una ejecución real**; nada se estima ni se rellena de memoria.

---

## 1. Principio

Se comparan seis implementaciones del **mismo contrato matemático** (`docs/model.md`, `docs/training.md`) sobre **el mismo hardware y las mismas condiciones**. Modalidad principal: **CPU, FP32, sin GPU, mono-hilo estricto**.

Decisión D-002 (aprobada): mono-hilo como categoría principal. La modalidad multi-hilo queda **definida pero desactivada** (§10) para una fase futura.

## 2. Dos clases de prueba, jamás mezcladas

| Clase | Qué valida | Herramienta | ¿Aparece en resultados de rendimiento? |
|---|---|---|---|
| **Corrección matemática** | Que el motor implementa el contrato: vectores dorados T1, paso de entrenamiento dorado T2, determinismo | `tests/` de cada motor + `shared/test_vectors/` | No. Se ejecutan **sin cronometrar**, en build DEBUG o RELEASE, antes de medir |
| **Rendimiento** | Tiempo, memoria y recursos del entrenamiento/inferencia reales | Arnés externo (§5) | Sí |

Orden obligatorio: un motor que no pasa la corrección no se mide. Un fallo de corrección invalida sus números de rendimiento.

## 3. Entorno controlado

Condiciones de la máquina de medición (la misma para los seis motores, misma sesión de medición):

1. **Un solo núcleo**: `taskset -c 0` en el lanzamiento del motor (afinidad de CPU fijada).
2. **Bibliotecas mono-hilo** (variables de entorno impuestas por el arnés):
   ```
   OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1  MKL_NUM_THREADS=1
   NUMEXPR_NUM_THREADS=1  VECLIB_MAXIMUM_THREADS=1
   ```
3. **Gobernador de CPU** en `performance` (si no se puede fijar, se documenta el valor real durante la sesión; no se silencia).
4. Sin otras tareas pesadas en el sistema; alimentación por corriente alterna; mismo estado térmico razonable (máquina en reposo previo).
5. **Sin fast-math** en las builds medidas (ver `docs/model.md` §9).
6. El arnés registra en cada sesión: modelo de CPU, frecuencia, RAM, kernel, versión de cada runtime/toolchain, fecha, y las variables anteriores (bloque `meta` del JSON de resultados, §8).

El arnés (`benchmark/harness/`, FASE 7) es **externo a los motores**: lanza cada ejecutable en subproceso con el entorno anterior y mide desde fuera. Así la metodología de medición es idéntica para los seis lenguajes y no contamina el código medido.

## 4. DEBUG y RELEASE

- **RELEASE (medida)**: build optimizada estándar de cada lenguaje (`-O3`/`-O2` para C/C++, `--release` para Rust, flags documentados para Java/Go). Sin `-ffast-math` ni equivalentes. Python no tiene build: se mide el intérprete CPython estándar con las versiones de Python/NumPy registradas; se documenta esta asimetría como inherente al lenguaje.
- **DEBUG (desarrollo)**: builds con assertions y símbolos para depurar. Nunca se miden.
- **Asimetrías documentadas**: C/C++/Rust compilan assertions fuera en RELEASE; Python las lleva siempre dentro. Se registra en `reports/comparison.md` como limitación conocida, no se oculta.

## 5. Procedimiento de ejecución

### 5.1 Corrección (previa, sin cronómetro)

1. T1: cada operación contra sus vectores dorados (`shared/test_vectors/`), tolerancias de `docs/architecture.md` §12.
2. T2: paso de entrenamiento dorado (lote 1, época 1) contra los valores de referencia, rel 1e-4.
3. Determinismo: dos ejecuciones completas → SHA-256 de `model_final.tb` idéntico.

### 5.2 Rendimiento

| Medición | Procedimiento | Repeticiones |
|---|---|---|
| Entrenamiento completo | 20 épocas + validaciones; se cronometra época a época y lote a lote | **5 ejecuciones** + calentamiento |
| Calentamiento | 1 época de entrenamiento descartada antes de cada ejecución medida (imports perezosos, caché de páginas, rampa de frecuencia) | Antes de cada ejecución |
| Latencia de inferencia | Cargar `model_final.tb`, clasificar una imagen (lote = 1), repetido | 100 llamadas de calentamiento + **1000 medidas** |
| Throughput de inferencia | Lote de 32 imágenes ya cargadas, forward repetido | 100 calentamiento + **100 medidas** |

- **Valores atípicos**: no se eliminan en silencio. Se reportan las 5 ejecuciones completas; la métrica central es la **mediana**; se calculan también media, desviación estándar y rango. Si una ejecución se desvía más de 2σ de la mediana, se marca `outlier: true` en el JSON y se documenta la causa sospechada (p. ej., interrupción del sistema). La comparación entre motores usa la mediana.
- Si la desviación estándar relativa del tiempo total supera el 5 %, se repite la sesión completa con 5 ejecuciones nuevas y se reportan ambas (regla anti-ruido).

### 5.3 Definición exacta de cada métrica temporal

| Métrica | Definición |
|---|---|
| `train_total_s` | Tiempo total de las 20 épocas (bucles de entrenamiento), **excluyendo** validación, checkpoints y logs en disco |
| `epoch_time_s` | Una época completa (250 lotes) |
| `batch_time_s` | Un lote: cargar índices + normalizar + forward + backward + actualización. Incluye la lectura del lote (uint8→FP32/255); **no** incluye escritura de logs |
| `val_epoch_s` | Una pasada de validación (1000 imágenes) |
| `inference_latency_s` | Una imagen individual (batch 1), modelo ya cargado, incluyendo su normalización |
| `inference_throughput_img_s` | Imágenes por segundo en lotes de 32 (forward solo) |
| `batch_p95_s` | Percentil 95 de los 5.000 `batch_time_s` (250×20) de la ejecución mediana |
| `inference_p95_s` | Percentil 95 de las 1000 latencias |

Reloj: `monotonic_ns` del arnés para tiempos externos (total, época); contador interno del motor para lote/inferencia (resolución microsegundo, en segundos con 6 decimales). Ambos se registran.

## 6. Las cinco categorías

### 6.1 Rendimiento

`train_total_s`, `epoch_time_s` (mediana por época), `batch_time_s` (mediana y p95), `inference_latency_s` (mediana y p95), `inference_throughput_img_s`. Todo en las condiciones de §3, 5 ejecuciones, mediana como métrica central.

### 6.2 Recursos

Medidos por el arnés muestreando `/proc/<pid>/` cada 100 ms (mismo método para todos los motores):

| Métrica | Fuente |
|---|---|
| `ram_max_mb`, `ram_mean_mb` | Pico y media de VmRSS |
| `cpu_mean_pct`, `cpu_max_pct` | (utime+stime) deltas entre muestras / tiempo muestreado (mono-hilo ⇒ ≈100 % si es CPU-bound) |
| `executable_size_mb` | Binario distribuido (C/C++/Rust/Go) o tamaño del venv necesario en ejecución (Python/Java con dependencias). Se documenta la asimetría |
| `model_size_mb` | Tamaño de `model_final.tb` (idéntico por contrato en todos) |

### 6.3 Calidad

Calculadas **una sola vez** por el evaluador común (FASE 7) a partir de `test_logits.csv` + `test_labels.bin` de cada motor (el cálculo de métricas no forma parte del código medido):

- `accuracy`, `precision`, `recall`, `f1` (umbral logit ≥ 0).
- `auc_roc`.
- `bce_test` (pérdida BCE media sobre test).
- Matriz de confusión 2×2.
- Calibración: 10 bines de probabilidad equiespaciados, por-bin accuracy medio vs confianza, y `ece` (error de calibración esperado).
- **Error numérico frente a la referencia**: contra la ejecución de referencia Python (`max_abs_logit_diff`, `mean_abs_logit_diff`, % de predicciones que cambian de etiqueta). Referencia = `reports/reference/python_reference.json`, generada por la primera ejecución Python validada.

### 6.4 Complejidad

| Métrica | Método |
|---|---|
| Líneas de código | `cloc` sobre el código del motor (src, excluye tests y arnés); se reporta total y por desglose |
| Tiempo de desarrollo | Registro real de horas por fase, por motor (registro de sesiones, no estimaciones) |
| Número de dependencias | Dependencias directas en runtime (sin contar toolchain); se listan con versión |
| Complejidad de gestión de memoria | Rubrica 1–5 con justificación escrita |
| Dificultad de depuración | Rubrica 1–5 con justificación escrita |
| Facilidad de optimización | Rubrica 1–5 con justificación escrita |
| Complejidad de paralelización | Rubrica 1–5 con justificación escrita |

Las rúbricas cualitativas las valora Adrián conmigo motor a motor, con evidencia citada (commits, incidencias concretas); son opinión documentada, no dato objetivo.

### 6.5 Portabilidad

| Métrica | Método |
|---|---|
| Facilidad de compilación | Número real de comandos de build/run desde un sistema limpio (documentado por motor) |
| Compatibilidad con SO | Compila/ejecuta en Linux (obligatorio); se registran intentos en otros SO si se prueban |
| Dependencias externas | Conteo y peso de la cadena de herramientas completa |
| Distribución | Facilidad de entregar el ejecutable a otra máquina igual (rubrica + descripción) |
| Ecosistema | Disponibilidad de bibliotecas útiles para este dominio (rubrica + justificación) |

## 7. Criterios de validez de una implementación

Una implementación es **válida para el benchmark** si:

1. Pasa T1 (vectores dorados de cada operación) con las tolerancias contractuales.
2. Pasa T2 (paso de entrenamiento dorado) con rel 1e-4.
3. Determinismo: dos ejecuciones producen el mismo SHA-256 en `model_final.tb`.
4. Completa las 5 ejecuciones de rendimiento sin errores ni NaN.
5. Sus artefactos de salida (`train_log.jsonl`, `test_logits.csv`, checkpoints) son conformes al formato de `docs/training.md` §5–7.
6. Cumple T3 (banda estadística de §8 de `docs/training.md`) o tiene una investigación documentada de la divergencia.

Si un motor falla 1–2, es **matemáticamente inválido** y no se publica su rendimiento. Si falla 3, es **no determinista** y se investiga antes de medir.

## 8. Formato de resultados

Estructura de ficheros:

```
benchmark/results/<lenguaje>/perf_run_001.json … perf_run_005.json
benchmark/results/<lenguaje>/resources_run_001.json …
benchmark/results/<lenguaje>/summary.json
reports/reference/python_reference.json
reports/<lenguaje>/informe.md
reports/comparison.md                     (comparativa consolidada, FASE 8)
```

Esquema `summary.json` (un fichero por lenguaje, agregado por el arnés):

```json
{
  "meta": {
    "engine": "python", "language_version": "3.14.9", "numpy_version": "2.5.4",
    "cpu_model": "...", "ram_gb": 62, "kernel": "...", "governor": "performance",
    "affinity": "core 0", "blas_threads": 1, "date": "2026-11-10T12:00:00+01:00",
    "build": "release", "fast_math": false
  },
  "performance": {
    "train_total_s": {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 5},
    "epoch_time_s": {"median": 0.0, "per_epoch_median": [0.0], "std": 0.0},
    "batch_time_s": {"median": 0.0, "p95": 0.0, "n": 5000},
    "val_epoch_s": {"median": 0.0, "std": 0.0},
    "inference_latency_s": {"median": 0.0, "p95": 0.0, "n": 1000},
    "inference_throughput_img_s": {"median": 0.0, "p95": 0.0, "n": 100}
  },
  "resources": {
    "ram_max_mb": 0.0, "ram_mean_mb": 0.0, "cpu_mean_pct": 0.0, "cpu_max_pct": 0.0,
    "executable_size_mb": 0.0, "model_size_mb": 0.3158
  },
  "quality": {
    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_roc": 0.0,
    "bce_test": 0.0, "confusion_matrix": [[0,0],[0,0]],
    "calibration_ece": 0.0, "calibration_bins": [[0.0,0.0]],
    "numeric_error_vs_reference": {"max_abs_logit_diff": 0.0, "mean_abs_logit_diff": 0.0, "flipped_predictions_pct": 0.0}
  },
  "complexity": {
    "loc_src": 0, "loc_total": 0, "dev_hours": 0.0, "runtime_dependencies": 0,
    "memory_mgmt": 0, "debug_difficulty": 0, "optimization_ease": 0, "parallelization": 0
  },
  "portability": {
    "build_commands": 0, "os_tested": ["linux"], "toolchain_size_mb": 0.0,
    "distribution": 0, "ecosystem": 0
  },
  "validity": {"t1_pass": true, "t2_pass": true, "deterministic": true, "t3_pass": null}
}
```

- Duraciones en segundos con 6 decimales; fracciones y probabilidades en [0,1] con 9 decimales.
- Los informes humanos (`reports/*.md`) se generan **desde** estos JSON; nunca se editan a mano los números.

## 9. Registro de sesiones

Cada sesión de medición escribe `benchmark/results/<lenguaje>/session.json` con: fecha/hora, temperatura de CPU antes/después (si `sensors` está disponible), carga del sistema previa, operador y notas. Sin este fichero la sesión no es publicable.

## 10. Modalidad multi-hilo (definida, desactivada)

Categoría secundaria futura (no parte de la comparación inicial): mismas métricas sin `taskset` y con threads libres, documentando por motor cuántos hilos usa y cómo paraleliza. Cada motor declara su estrategia de paralelización. Se activará como extensión cuando existan al menos dos motores compilados; los resultados se publican en sección separada para no contaminar la comparación mono-hilo.

## 11. Cronograma de medición

1. FASE 7-A: arnés + primera sesión completa Python → genera `reports/reference/python_reference.json`.
2. FASE 7-B (futuro, por motor): corrección → 5 ejecuciones → summary → informe.
3. FASE 8: `reports/comparison.md` consolidada con las cinco categorías y las limitaciones encontradas.

## 12. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción completa (FASE 0). |
