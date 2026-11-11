# decisions.md — Registro de decisiones y pendientes de aprobación

> **Estado**: FASE 0. Se actualiza con cada decisión tomada.
> Reglas: toda decisión relevante se registra aquí con número, contexto, alternativas y consecuencia. Las decisiones **D-XXX** están tomadas; las **P-XXX** requieren aprobación de Adrián antes de avanzar. Cambios de contrato → bump de versión en los documentos afectados (`architecture.md` §11).

---

## A. Decisiones tomadas

### D-001 — Contrato matemático + estrategia libre por motor
**Contexto**: la comparación entre lenguajes exige que nadie obtenga ventaja de una arquitectura distinta, pero también quiere medir lo que cada ecosistema logra realmente.
**Decisión** (aprobada por Adrián, 2026-11-10): el contrato (`docs/model.md`) fija la **función matemática exacta**, formas y convenciones. Cada motor elige su **estrategia interna** (im2col+GEMM, bucles directos, etc.) y la documenta en su informe. Las tolerancias de `architecture.md` §12 garantizan la equivalencia.
**Alternativas descartadas**: algoritmo canónico único (mediría el intérprete de Python en convoluciones, no el sistema); doble implementación canónica+optimizada (duplicaría LOC y el trabajo en 6 lenguajes).
**Consecuencia**: el LOC y la dificultad difieren legítimamente entre motores; queda registrado como parte de lo que se estudia.

### D-002 — Mono-hilo estricto como categoría principal del benchmark
**Decisión** (aprobada): CPU, FP32, sin GPU, 1 hilo (`taskset -c 0`, variables BLAS a 1, gobernador performance). Modalidad multi-hilo definida pero desactivada (`benchmark.md` §10).
**Consecuencia**: comparación núcleo a núcleo, reproducible y coherente con la modalidad declarada. El paralelismo se estudia en fase posterior sin contaminar los resultados iniciales.

### D-003 — Selección estratificada determinista con permutación canónica
**Decisión** (aprobada): cuotas exactas por clase (reparto de restos por orden ascendente de id) + Fisher-Yates con splitmix64 documentado, semillas fijas. Los motores cargan manifiestos, no regeneran (`dataset.md` §5).
**Consecuencia**: composición equilibrada entre las 9 clases negativas y reproducibilidad total sin PRNG en runtime.

### D-004 — Salida en logit + BCE con logits numéricamente estable
**Decisión** (de la especificación de Adrián, formalizada): la red produce un logit; la pérdida consume logits con la forma `max(z,0) − z·y + log(1+e^{−|z|})`; Sigmoid solo en inferencia/métricas. El gradiente analítico combinado `dZ = (σ(z) − y)/N` inicia el backward.

### D-005 — Sin BatchNorm, Dropout ni capas extra
**Decisión** (de la especificación): arquitectura exactamente las 10 capas de `model.md` §3. Cualquier capa adicional exige una decisión nueva con justificación y actualización de contrato.

### D-006 — Artefactos de referencia generados una vez por la herramienta Python
**Decisión**: pesos iniciales, permutaciones y vectores de prueba se generan una única vez con semillas registradas (maestra `20261110`) y viven en `shared/` en formatos independientes del lenguaje. Ningún motor genera aleatoriedad.
**Detalle**: generador de normales para He/Xavier → D-018.

### D-007 — Permutaciones de época pregeneradas como ficheros
**Decisión**: 20 ficheros `.perm` (uint32 LE) en `shared/permutations/`. El orden de presentación de muestras es idéntico en los seis motores sin implementar PRNG.
**Alternativa descartada**: PRNG implementado por cada motor (riesgo de divergencia por diferencias sutiles de implementación; complejidad extra sin beneficio para la comparación).

### D-008 — Convenciones de borde: ReLU(0) y empates de MaxPool
**Decisión**: derivada de ReLU en 0 = 0 (subgradiente). Empate en MaxPool: el gradiente va al primer máximo en orden raster. Ambas reglas son trazales a cualquier lenguaje (evitan ambigüedad y son reproducibles).

### D-009 — Dataset derivado uint8 crudo + normalización bit-exacta en carga
**Decisión**: `shared/dataset/*.bin` en uint8 CHW (mismos bytes que el original, reshape puro). Cada motor normaliza `float32(u8)/255.0` al cargar: conversión y división son operaciones IEEE exactas → entradas bit-idénticas en los seis lenguajes sin almacenar FP32.

### D-010 — Particiones y modelo final sin selección
**Decisión**: validation sale del pool de entrenamiento; test solo del pool de test; sin early stopping ni selección de mejor época: el modelo evaluado es el de la época 20. Evita comparar motores con criterios de selección distintos.

### D-011 — Independencia estricta de datos de cada motor
**Decisión**: ningún motor lee el tar.gz original, pickle, HTTP ni manifiesto de descarga. Solo la herramienta Python (`scripts/`). El motor Python tampoco puede leer pickle (por simetría contractual con los otros cinco).

### D-012 — Métricas de calidad calculadas por un evaluador común
**Decisión**: los motores solo escriben `test_logits.csv`; precision/recall/F1/AUC/calibración las calcula `scripts/evaluate.py` idéntico para todos. El cálculo de métricas no forma parte del código medido.

### D-013 — Arnés de medición externo
**Decisión**: `benchmark/harness/` lanza cada motor en subproceso con entorno controlado y mide desde fuera (`/proc`, reloj monotónico). Metodología de medición idéntica para los seis.

### D-014 — Entrada de inferencia cruda y esquema JSON
**Decisión**: `--image`/`--batch` aceptan binarios uint8 de 3072·N bytes (layout del dataset). Sin decodificación PNG/JPEG en ningún motor. Salida JSON `{prediction, probability, confidence}` con `confidence = max(σ(z), 1−σ(z))`, umbral logit ≥ 0.

### D-015 — Reglas de coma flotante
**Decisión**: FP32 en todo el grafo; FMA permitido (cubierto por tolerancias); `-ffast-math` y equivalentes prohibidos en las builds medidas; NaN/Inf abortan el entrenamiento. Determinismo bit a bit exigido dentro de cada motor (misma máquina, mismo modo).

### D-016 — Formato de intercambio de tensores: TensorBundle v1
**Decisión**: formato binario little-endian autodescrito con índice por nombre (`architecture.md` §5) para pesos, checkpoints y vectores de prueba. Legible en C puro sin dependencias.

### D-017 — Política de git de artefactos
**Decisión**: se suben a git los artefactos contrato **congelados** (pesos iniciales y vectores de prueba, ~4 MB con checksums) para que el contrato no dependa de regeneraciones; quedan fuera los binarios grandes regenerables de forma bit-exacta por PRNG propio (dataset `*.bin`, `*.perm`, tar.gz) y las salidas de motores (`runs/`).

### D-018 — Generador de normales: Box-Muller sobre splitmix64 *(antes P-001; aprobada 2026-11-11)*
Box-Muller clásico, un solo flujo (`x₀ = 20261110 + 1000`), orden `conv1`, `conv2`, `dense1`, `dense2`. Detalle exacto (conversión a u1/u2, FP64 y cast) en `architecture.md` §7.1. Desbloquea la FASE 2.

### D-019 — Validación por lotes de 32 con último lote parcial *(antes P-002; aprobada)*
### D-020 — Banda T3 provisional: `val_accuracy` ±0,03 y `val_loss` ±10 % *(antes P-003; aprobada)*
Se revisará con los dos primeros motores reales; antes de tocar la banda se investigan T1/T2 (`training.md` §9).
### D-021 — Benchmark: 5 ejecuciones + 1 época de calentamiento + 1000 latencias *(antes P-004; aprobada)*
Con regla anti-ruido (std relativa > 5 % → repetir sesión). La semántica exacta del calentamiento queda en P-014.
### D-022 — Motor Rust: implementación nueva, sin reutilizar `rust-ai-core` *(antes P-005; aprobada)*
### D-023 — BLAS de NumPy: registrar, no restringir *(antes P-006; aprobada)*
### D-024 — Repositorio local, sin remoto *(antes P-007; **por defecto**, sin recomendación explícita)*
Reversible en cualquier momento; si quieres remoto (GitHub privado/público) dilo y se añade.
### D-025 — Tiempo de desarrollo: registro manual por sesión en el informe de cada motor *(antes P-008; aprobada)*
### D-026 — Código: identificadores en inglés, comentarios y docstrings en español *(antes P-009; aprobada)*
### D-027 — `cifar10.py` pasa a `scripts/` (ningún motor lee pickle) *(antes P-010; aprobada)*

### D-028 — CIFAR-10 en formato binario *(antes P-011; aprobada 2026-11-11)*
`cifar-10-binary.tar.gz`, MD5 `c32a1d4ab5d03f1284b67883e8d87530`. Sin pickle en ningún punto del proyecto. SHA-256 pendiente de fijar en la descarga (FASE 4).
### D-029 — Selección estratificada con un único flujo splitmix64 *(antes P-012; aprobada)*
Orden: pool de entrenamiento clases 0→9, luego pool de test clases 0→9 (`dataset.md` §5.1).
### D-030 — Atípicos por MAD (3·1,4826·MAD) o 10 % de desviación *(antes P-013; aprobada)*
### D-031 — Calentamiento en el mismo proceso con recarga de pesos iniciales *(antes P-014; aprobada)*
### D-032 — Datos enteros a RAM antes de cronometrar; `data_load_s` aparte *(antes P-015; aprobada)*

### D-033 — Vectores dorados calculados en FP64 y guardados como FP32 *(2026-11-11)*
**Contexto**: `architecture.md` §10.2 exigía generarlos «desde operaciones validadas»; faltaba decir en qué precisión.
**Decisión**: referencia en FP64 sobre entradas FP32 exactas. **Alternativa descartada**: generarlos con el propio motor FP32 (la prueba T1 compararía el motor consigo mismo).
**Consecuencia**: la tolerancia 1e-5/1e-4 mide el error real del motor FP32; margen medido en `docs/python.md` §10.

### D-034 — `train_step`: caso sintético ahora, lote real 1 en la FASE 4 *(2026-11-11)*
**Contexto**: `architecture.md` §10.1 define el caso T2 con el lote real 1 de la época 1, que necesita el dataset y las permutaciones (FASE 4).
**Decisión**: `train_step/case_001` usa un lote **sintético** de 32 (splitmix64 → uint8 → `/255`) con los pesos iniciales compartidos; `case_002` (lote real) se añade en la FASE 4. Así la matemática del paso completo queda verificable ya, sin esperar al dataset.
**Consecuencia**: el criterio de aceptación 6 de `training.md` §10 se cumple solo cuando existan ambos casos.

### D-037 — Detalles de la inferencia no fijados por el contrato *(2026-11-11; lagunas cerradas en la FASE 6)*
- La clase negativa se etiqueta `"NOT_DOG"` (nombre ya usado en model.md y dataset.md; architecture.md §8 solo ejemplificaba `"DOG"`).
- `probability` y `confidence` se calculan con σ en FP32, se pasan a FP64 y se redondean a 3 decimales con `round` de Python (mitades al par); `confidence = max(p, 1−p)` sobre el valor sin redondear.
- `--image` exige exactamente 3072 bytes; `--batch` un múltiplo no nulo de 3072. Cualquier entrada inválida, fichero inexistente o modelo corrupto → mensaje en stderr y código de salida 2, sin salida JSON.
- Se pasa `--image` o `--batch`, nunca ambos ni ninguno. Salida JSON compacta en una línea (objeto con `--image`, array con `--batch`).
- El modelo puede ser `model_final.tb`, un checkpoint o `initial_weights.tb` (se ignoran los `meta.*`).

### D-036 — Detalles del entrenamiento no fijados por el contrato *(2026-11-11; lagunas cerradas en la FASE 5)*
- `train_loss` de época y `val_loss`/`test_loss` se calculan como media en **FP32 con suma secuencial** en el orden de presentación/manifiesto (training.md §4-§5 solo decía «media»).
- `meta.contract_version` = **1** (contrato v1.0), el mismo valor que `manifest.csv` de los vectores.
- `main.py train` rechaza un directorio de ejecución ya usado (no sobrescribe resultados) y fija BLAS/OpenMP a 1 hilo (D-002) antes de importar NumPy.
- `batch_log.csv` (opcional en training.md §5) no se implementa por ahora.

### D-035 — Orden de las muestras dentro de cada conjunto *(2026-11-11; laguna del contrato cerrada)*
**Contexto**: `dataset.md` v1 definía qué muestras entran en cada conjunto pero no su orden en los ficheros (afecta a las posiciones de las permutaciones, al orden de validación/test y al lote T2 real).
**Decisión**: orden ascendente de índice global (fuente y luego índice). **Alternativa descartada**: orden de selección (barajado por clase): menos auditable y sin ventaja.

---

## B. Decisiones pendientes de aprobación

### P-016 — Criterio de error escalado además de atol/rtol en T1/T2 *(afecta a los 5 motores futuros)*
**Hallazgo (FASE 3)**: las tolerancias de `architecture.md` §12 (`atol = rtol = 1e-4` en gradientes y pesos) son 100–1000× más laxas que el error real del motor FP32 (≈1e-7). El 60 % de `dense1.weight` y el 34 % de `conv2.weight` valen < 1e-4 en el lote T2, y `lr·g ≤ 1e-2`. Medido con mutaciones: **un `lr` erróneo del 1 % o 0,1 % pasa T2 con el contrato actual**.
- **Recomendada**: añadir un criterio por tensor `max|a−b| / max|b| < 1e-5` (error escalado; valor medido ≤ 1,1e-6, margen 9×) como **segundo requisito** de T1/T2, y mantener atol/rtol tal cual. Un motor con peor aritmética (p. ej. suma ingenua) tendría que justificar y documentar el incumplimiento.
- Alternativa: no tocar el contrato; el motor Python ya lo aplica en sus propias pruebas (`test_scaled_error_per_tensor`, `test_t2_weight_update_is_exactly_lr_times_gradient`) pero los otros cinco quedarían con un T2 débil.
- No bloquea la FASE 4.

## C. Cuadro de fases y estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Análisis y especificación (4 contratos + architecture + python + decisions) | Hecha (2026-11-10) |
| 1 | Validación documental cruzada | **Hecha 2026-11-11** |
| 2 | Motor matemático Python + vectores dorados | **Hecha 2026-11-11** (operaciones; el modelo completo es la FASE 3) |
| 3 | Integración del modelo | **Hecha 2026-11-11** (P-016 abierto, no bloquea) |
| 4 | Dataset (descarga, derivación, manifiestos) | **Hecha 2026-11-11** |
| 5 | Entrenamiento completo | Hecha (2026-11-11) |
| 6 | Inferencia | Hecha (2026-11-11) |
| 7 | Arnés de benchmark + línea base Python | Pendiente |
| 8 | Documentación final | Pendiente |

## Historial del registro

| Fecha | Cambio |
|---|---|
| 2026-11-10 | Creación del registro con D-001…D-017 y P-001…P-010. |
| 2026-11-11 | P-001…P-010 aprobadas (P-007 por defecto) → D-018…D-027. FASE 1: P-011…P-015 nuevos. |
| 2026-11-11 | P-011…P-015 aprobadas → D-028…D-032. Sin pendientes. |
| 2026-11-11 | FASE 2 hecha; D-033. |
| 2026-11-11 | FASE 3 hecha; D-034 y P-016. |
| 2026-11-11 | FASE 4 hecha; D-035. P-016 sigue pendiente. |
| 2026-11-11 | FASE 5 hecha; D-036. P-016 sigue pendiente. |
| 2026-11-11 | FASE 6 hecha; D-037. P-016 sigue pendiente. |
