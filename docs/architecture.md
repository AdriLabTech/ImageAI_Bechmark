# architecture.md — Contratos compartidos entre las seis implementaciones

> **Estado**: FASE 0 (especificación). Versión 1.0-draft. Pendiente de aprobación.
> Este documento define **cómo se relacionan las seis implementaciones**: qué contratos comparten, qué formatos de ficheros intercambian, qué tolerancias exigen y qué papel juega cada pieza del repositorio. Es el documento puente entre `docs/model.md` (matemática) y las implementaciones.

---

## 1. Propósito

Seis motores independientes (Python, C, Rust, C++, Java, Go) implementando el mismo contrato matemático (`model.md`), entrenando con el mismo procedimiento (`training.md`) y midiéndose con el mismo arnés (`benchmark.md`). Ningún motor necesita ejecutar a otro. La coordinación ocurre **a través de artefactos en disco**, no de llamadas entre programas.

## 2. Estructura del repositorio

```
ImageAI-Benchmark/
├── docs/                     # Contratos (este documento y sus hermanos)
│   ├── model.md             # Contrato matemático de la red
│   ├── dataset.md            # Contrato de datos y particiones
│   ├── training.md           # Contrato del ciclo de entrenamiento
│   ├── benchmark.md          # Contrato de medición
│   ├── architecture.md       # Este documento: contratos entre motores
│   ├── python.md             # Propuesta de arquitectura del motor Python
│   └── decisions.md          # Registro de decisiones y pendientes
│
├── scripts/                  # Herramientas compartidas (solo Python, no se miden)
│   ├── download_cifar10.py   # FASE 4
│   ├── prepare_dataset.py    # FASE 4: dataset derivado + manifiestos + permutaciones
│   ├── generate_artifacts.py # FASE 2: pesos iniciales + vectores de prueba
│   └── evaluate.py           # FASE 7: métricas de calidad comunes
│
├── implementations/          # Motores medibles (uno por lenguaje)
│   └── python/               # FASE 2+: el primero
│   ├── c/                    # futuro
│   ├── rust/                 # futuro
│   ├── cpp/                  # futuro
│   ├── java/                 # futuro
│   └── go/                   # futuro
│
├── shared/                   # Artefactos canónicos intercambiables (binarios grandes en .gitignore, D-017)
│   ├── dataset/              # *.bin, manifests/, CHECKSUMS.sha256
│   ├── initial_weights/      # initial_weights.tb
│   ├── permutations/         # train_epoch_NNN.perm
│   └── test_vectors/         # vectores dorados por operación
│
├── benchmark/                # Arnés de medición + resultados crudos
│   ├── harness/              # FASE 7
│   └── results/<lenguaje>/
│
├── reports/                  # Informes generados (humano)
│   ├── reference/            # python_reference.json (línea base)
│   └── <lenguaje>/informe.md
│
├── tests/                    # Pruebas transversales del repositorio
└── README.md
```

Reglas:

- `shared/` es la **única** vía de intercambio entre motores. Contiene artefactos binarios; los grandes y regenerables (`dataset/*.bin`, `*.perm`) no se suben a git (`dataset.md` §11) y los pesos iniciales y vectores dorados sí (D-017).
- Cada motor escribe sus salidas propias en `implementations/<lenguaje>/runs/` (gitignored): checkpoints, `train_log.jsonl`, `test_logits.csv`. El arnés los recoge hacia `benchmark/results/`.
- Los motores futuros se añaden creando su carpeta: nada de la estructura actual necesita reorganizarse.

## 3. Roles y separación

| Rol | Código | ¿Se mide? | Lenguaje | Reglas |
|---|---|---|---|---|
| **Herramienta compartida** | `scripts/` | No | Python | Puede usar cualquier biblioteca Python razonable. Genera y regenera artefactos. Es la única pieza que toca la fuente original de datos |
| **Motor** | `implementations/<lenguaje>/` | Sí | Uno por lenguaje | Implementa el contrato completo con las restricciones de `docs/python.md` (o equivalente por lenguaje). No descarga, no lee la fuente original, no genera azar |
| **Arnés de medición** | `benchmark/harness/` | No (mide) | Python | Lanza motores en subproceso con entorno controlado (`benchmark.md` §3). Mide desde fuera para que la metodología sea idéntica para todos |
| **Evaluador de calidad** | `scripts/evaluate.py` | No | Python | Calcula métricas de calidad desde `test_logits.csv` de cada motor, con el mismo código para todos |

**Python tiene doble papel** (herramienta y motor #1) y por eso vive en dos sitios: `scripts/` (herramienta, sin restricciones) e `implementations/python/` (motor, medido y restringido). El motor Python nunca importa de `scripts/`.

## 4. Contratos conceptuales (interfaces lenguaje-agnóstico)

Estas interfaces se descricen sin rasgos exclusivos de Python. Cada lenguaje las implementa con sus propios mecanismos (clases, structs con punteros a función, traits, interfaces…).

### 4.1 Operación / capa

```
Operación:
  forward(entrada)        → salida          # valida dimensiones; error explícito si no encajan
  backward(grad_salida)   → grad_entrada    # y además calcula grad_params si tiene parámetros
  params()                → [(nombre, tensor)]      # parámetros entrenables (vacío si no tiene)
  grads()                 → [(nombre, tensor)]       # mismas formas que params()
  zero_grads()            → reinicia gradientes
```

- `backward` calcula el gradiente **total del lote** y lo **almacena sobrescribiendo** (no acumula entre lotes: el optimizador consume tras cada lote, `training.md` §3).
- La validación de dimensiones es obligatoria en `forward`: un motor que acepte silenciosamente formas incorrectas no es válido.
- Las operaciones sin parámetros (ReLU, MaxPool, Flatten) devuelven lista vacía en `params()`.

### 4.2 Modelo

```
Modelo:
  forward(lote)           → logits          # docs/model.md §5
  backward(dZ)            → recorre capas en orden inverso (docs/model.md §6)
  params()/grads()/zero_grads()             # agregados de las capas
  save(ruta) / load(ruta)                   # TensorBundle con nombres canónicos
```

Nombres canónicos de tensores (idénticos en ficheros y en todos los motores): `conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `dense1.weight`, `dense1.bias`, `dense2.weight`, `dense2.bias`.

### 4.3 Entrenador e inferencia

```
Entrenador:
  train()                 # ejecuta docs/training.md §3 exacto
  # produce: checkpoints por época, train_log.jsonl, test_logits.csv, model_final.tb

Predictor:
  load(ruta_modelo)
  predict(imagen)          → {"prediction", "probability", "confidence"}   # §8
  predict_batch(lote)      → [ ... ]
```

El predictor no depende del entrenador: carga `model_final.tb` y clasifica sin tocar el ciclo de entrenamiento.

## 5. Formato TensorBundle v1 (`.tb`)

Formato binario para **cualquier colección de tensores con nombre**. Es el formato de: pesos iniciales, checkpoints, vectores de prueba. Diseñado para poder leerse en C puro sin dependencias.

**Convenciones globales**: little-endian; offsets en bytes desde el inicio del fichero; sin compresión.

### 5.1 Cabecera (12 bytes)

| Offset | Tamaño | Campo |
|---|---|---|
| 0 | 4 | Magic: `41 49 54 42` (ASCII «AITB») |
| 4 | 2 | `version`: uint16 = 1 |
| 6 | 2 | `flags`: uint16 = 0 (reservado; ignorar si 0) |
| 8 | 4 | `n_tensors`: uint32 |

### 5.2 Índice (n_tensors entradas de 72 bytes)

| Offset | Tamaño | Campo |
|---|---|---|
| +0 | 32 | `name`: ASCII con terminador `\0` y relleno (máx. 31 caracteres) |
| +32 | 2 | `dtype`: uint16 (tabla siguiente) |
| +34 | 1 | `ndim`: uint8 ∈ [1, 4] |
| +35 | 1 | reservado = 0 |
| +36 | 4 | reservado = 0 |
| +40 | 8 | `data_offset`: uint64 |
| +48 | 8 | `n_elements`: uint64 (debe coincidir con el producto de `shape`) |
| +56 | 16 | `shape`: ndim × uint32 (máx. 4 dimensiones = 16 B); las posiciones ≥ ndim quedan a 0 |

### 5.3 Códigos de dtype

| Código | Tipo | Bytes/elemento |
|---|---|---|
| 0 | f32 (IEEE 754 binary32) | 4 |
| 1 | f64 (IEEE 754 binary64) | 8 |
| 2 | u8 | 1 |
| 3 | i32 | 4 |
| 4 | u32 | 4 |
| 5 | i16 | 2 |

### 5.4 Reglas de datos

- Los bloques de datos empiezan tras el índice (`12 + 72·n_tensors`).
- Cada tensor se escribe en row-major (orden C), su `data_offset` **debe** ser múltiplo de 4 y el bloque se rellena (padding) hasta múltiplo de 4; el lector ignora el relleno.
- `n_elements = Π shape[i]`; un lector debe validar esta coherencia y fallar en caso contrario.
- Los tensores del índice aparecen en orden alfabético de `name` (regla de escritura; el lector no debe asumir orden, debe buscar por nombre).

### 5.5 Metadatos en checkpoints

Sin JSON: los metadatos son tensores escalares con nombres reservados, legibles en cualquier lenguaje:

| Nombre | dtype | shape | Contenido |
|---|---|---|---|
| `meta.epoch` | u32 | [1] | Época del checkpoint |
| `meta.train_loss` | f32 | [1] | Pérdida media de la época |
| `meta.val_loss` | f32 | [1] | Pérdida de validación |
| `meta.val_accuracy` | f32 | [1] | Exactitud de validación |
| `meta.contract_version` | u32 | [1] | Versión del contrato (§11) |

## 6. Otros formatos compartidos (resumen)

| Artefacto | Formato | Especificación completa |
|---|---|---|
| `shared/dataset/*_images.bin`, `*_labels.bin` | uint8 crudo, layout plano CHW | `dataset.md` §6 |
| `shared/permutations/train_epoch_NNN.perm` | uint32 LE crudo (8000 entradas) | `dataset.md` §8 |
| `shared/dataset/manifests/*.csv` | CSV UTF-8 con cabecera | `dataset.md` §7 |
| `train_log.jsonl`, `final_metrics.json` | JSON Lines / JSON | `training.md` §5 |
| `test_logits.csv` | CSV `position,logit` | `training.md` §7 |
| `summary.json` de benchmark | JSON | `benchmark.md` §8 |
| `CHECKSUMS.sha256` | Lista SHA-256 | `dataset.md` §10 |

Los ficheros `.perm` y `.bin` de dataset **no** usan TensorBundle deliberadamente: son crudos de tamaño fijo conocidos por convención de nombre, más pequeños y mapeables en memoria directamente.

## 7. Semillas y PRNG (registro)

| Elemento | Valor |
|---|---|
| Semilla maestra del proyecto | `20261110` |
| Selección estratificada del dataset | splitmix64, un solo flujo continuo, `x₀ = 20261110` (`dataset.md` §5.1) |
| Permutación de época `e` (1–20) | splitmix64 con `x₀ = 20261110 + e` |
| Pesos iniciales | splitmix64 con `x₀ = 20261110 + 1000` + Box-Muller (D-018, §7.1) |
| Azar en runtime de los motores | **Ninguno** |

El PRNG (splitmix64, pseudocódigo en `dataset.md` §5.2) solo existe dentro de `scripts/`. Ningún motor lo implementa: consume artefactos. Esto elimina la clase entera de bugs de divergencia por generadores distintos.

### 7.1 Generación de los pesos iniciales (D-018)

Un único flujo splitmix64 (`x₀ = 20261110 + 1000`, distinto de la selección y de las épocas 1–20) llena los tensores en este orden: `conv1.weight`, `conv2.weight`, `dense1.weight`, `dense2.weight`, cada uno en orden row-major. Los bias son 0 y no consumen flujo.

Cada par de elementos consecutivos consume **dos** salidas `a`, `b` del PRNG:

```
u1 = ((a >> 11) + 1) · 2⁻⁵³      # ∈ (0, 1], evita log(0)
u2 =  (b >> 11)      · 2⁻⁵³      # ∈ [0, 1)
r  = sqrt(−2 · ln u1)
z0 = r · cos(2π·u2),  z1 = r · sin(2π·u2)
elemento = float32(std · z)        # std de model.md §7, en FP64 antes del cast
```

Todos los tensores tienen un número par de elementos (432, 4.608, 73.728, 64), así que no sobra ningún `z`. El cálculo usa FP64 y `math.cos/log` de la herramienta Python; como el resultado se congela en `initial_weights.tb`, el fichero —no el algoritmo— es el contrato (una regeneración puede diferir en el último ulp según la libm).


## 8. Contrato de línea de comandos de cada motor

Cada motor expone tres comandos equivalentes (flags idénticos, independientes del lenguaje):

```
<motor> tests  --data-dir shared/                     # T1 + T2 + determinismo; exit 0/1
<motor> train  --data-dir shared/ --out runs/          # entrenamiento completo (training.md)
<motor> infer  --model model_final.tb --image img.bin # una imagen (3072 bytes uint8)
<motor> infer  --model model_final.tb --batch b.bin    # lote (N×3072 bytes uint8)
```

- `--image`/`--batch` aceptan **ficheros binarios crudos uint8** con el mismo layout que el dataset (§6 de `dataset.md`). Ningún motor decodifica PNG/JPEG: evita dependencias de imagen en C/Go y garantiza entradas idénticas. El proyecto no necesita decodificación de formatos de imagen.
- La salida de `infer` es JSON (UTF-8):

```json
{
  "prediction": "DOG",
  "probability": 0.987,
  "confidence": 0.987
}
```

Definiciones exactas: `prediction` = "DOG" si `logit ≥ 0` (umbral 0,5 en probabilidad); `probability` = σ(logit) = P(DOG), redondeada a 3 decimales; `confidence` = probabilidad de la clase predicha = `max(σ(z), 1−σ(z))`, redondeada a 3 decimales. En lotes: array JSON de objetos.

- `tests` imprime un resumen por operación con la violación máxima de tolerancia (no solo pass/fail) y termina con código distinto de cero si algo falla.

## 9. Flujo de datos entre componentes

```
        CIFAR-10 (URL oficial)
               │  scripts/download_cifar10.py (verifica MD5/SHA-256)
               ▼
        shared/dataset/raw/cifar-10-binary.tar.gz
               │  scripts/prepare_dataset.py  (único lector de la fuente original)
               ▼
   shared/dataset/*.bin + manifests/ + permutations/ + CHECKSUMS.sha256
               │  scripts/generate_artifacts.py (splitmix64, He/Xavier)
               ▼
   shared/initial_weights/ + shared/test_vectors/
               │
    ┌──────────┼──────────────┬─────────────┐
    ▼          ▼              ▼             ▼
 motor      motor rust     motor cpp   ... (6 motores, solo leen shared/)
    │          │              │
    └──────────┴──────┬───────┘
                      ▼   implementaciones/<lang>/runs/
        {model_final.tb, train_log.jsonl, test_logits.csv, checkpoints}
                      │  benchmark/harness (entorno controlado, cronómetro externo)
                      ▼
        benchmark/results/<lang>/{perf_runs, resources, summary}.json
                      │  scripts/evaluate.py (métricas idénticas para todos)
                      ▼
        reports/reference/python_reference.json + reports/comparison.md
```

## 10. Vectores de prueba (golden vectors)

### 10.1 Estructura

```
shared/test_vectors/
├── manifest.csv                    # case, operación, propósito, sha256 del .tb
├── conv2d/case_001.tb … case_004.tb
├── maxpool2d/case_001.tb … case_004.tb
├── relu/case_001.tb … case_002.tb
├── sigmoid/case_001.tb … case_002.tb
├── flatten/case_001.tb
├── dense/case_001.tb … case_002.tb
├── bce/case_001.tb … case_003.tb
├── model/case_001.tb               # forward completo con activaciones intermedias
└── train_step/case_001.tb          # T2: lote dorado + pesos tras una actualización
```

Cada `case_NNN.tb` es un TensorBundle con tensores con nombre según la operación:

| Operación | Tensores incluidos |
|---|---|
| conv2d | `input`, `weight`, `bias`, `output`, `grad_output`, `grad_input`, `grad_weight`, `grad_bias` |
| maxpool2d | `input`, `output`, `grad_output`, `grad_input` |
| relu / sigmoid | `input`, `output`, `grad_output`, `grad_input` |
| flatten | `input`, `output`, `grad_output`, `grad_input` |
| dense | `input`, `weight`, `bias`, `output`, `grad_output`, `grad_input`, `grad_weight`, `grad_bias` |
| bce | `input` (logits), `labels` (u8), `loss` (f32 escalar), `grad_input` |
| model | `input`, `a1`…`a9`, `logits` (activaciones intermedias para localizar divergencias) |
| train_step | `input`, `labels`, `weights_init.*` (8), `loss`, `grads.*` (8), `weights_updated.*` (8) |

Casos obligatorios por operación (mínimos; se añadirán si se detectan divergencias):

- **conv2d**: (1) formas reales de conv1 (1,3,32,32); (2) formas reales de conv2; (3) un caso con N=2 y C=1 pequeño para depurar a mano; (4) caso con valores negativos.
- **maxpool2d**: (1) entrada 30×30 par; (2) entrada **13×13 impar** (descarte); (3) caso con **empates** de máximo (regla del primer máximo en orden raster); (4) formas reales.
- **relu**: (1) con ceros y negativos exactos; (2) formas reales con valores extremos.
- **sigmoid**: (1) valores en [-30, 30] (estabilidad); (2) rango fino alrededor de 0.
- **dense**: (1) formas reales dense1; (2) caso cuadrado pequeño verificable a mano.
- **bce**: (1) logits = 0; (2) logits grandes positivos y negativos (sin overflow); (3) mezcla realista.
- **model**: batch de 4 imágenes sintéticas deterministas.
- **train_step**: el lote real 1 de la época 1 (posiciones `perm₁[0..32)` de train) con los pesos iniciales compartidos.

### 10.2 Generación y arranque (bootstrap) del sistema de confianza

1. La implementación Python de cada operación se valida primero con **gradient check por diferencias finitas en FP64** (`model.md` §10) — la matemática se demuestra contra el cálculo numérico, no contra sí misma.
2. Solo entonces `scripts/generate_artifacts.py` emite los vectores dorados **desde las operaciones ya validadas**.
3. El propio motor Python pasa T1 contra esos ficheros (autochequeo de serialización/deserialización incluido).
4. Los otros cinco motores se verifican contra los mismos ficheros sin ningún paso adicional.

Los vectores dorados son, por tanto, artefactos auditables generados por código verificado; no «la salida de Python» a ciegas.

## 11. Versionado de contratos y protocolo de cambio

- Los cuatro contratos + este documento forman la **versión de contrato** (v1.0 al aprobarse). Se registra en `meta.contract_version` de cada checkpoint y en `manifest.csv` de vectores de prueba.
- Cambio propuesto → se registra en `docs/decisions.md` (número D-XXX, motivación, alternativas) → se actualizan los documentos afectados → bump de versión → `scripts/generate_artifacts.py` regenera los artefactos afectados → cada motor actualiza y debe volver a pasar `tests`.
- Un motor declara en `summary.json.meta` la versión de contrato que implementa. No se comparan motores con versiones de contrato distintas.

## 12. Tolerancias numéricas

Comparación elemento a elemento contra el valor de referencia `b`:

```
aprobado  ⟺  |a − b| ≤ atol + rtol · |b|
```

| Magnitud comparada (T1/T2) | atol | rtol |
|---|---|---|
| Salidas forward de operaciones aisladas | 1e-5 | 1e-5 |
| Gradientes de operaciones aisladas | 1e-4 | 1e-4 |
| Pérdida (escalar) | 1e-5 | 1e-5 |
| Logits del modelo completo (caso `model`) | 1e-4 | 1e-4 |
| Pesos tras un paso de entrenamiento (T2) | 1e-4 | 1e-4 |
| Activaciones intermedias del caso `model` | 1e-4 | 1e-4 |

Justificación: epsilon de FP32 ≈ 1,19·10⁻⁷; una acumulación de ~10³–10⁴ términos (convoluciones sobre un lote) acumula errores relativos esperados de ~10⁻⁶–10⁻⁵. Las bandas dejan margen para diferencias legítimas de orden de suma (incluido FMA) y siguen detectando errores reales de implementación, que típicamente superan 10⁻². Reglas adicionales:

- Un NaN o Inf en cualquier valor comparado es fallo inmediato, no sujeto a tolerancia.
- Cada comparación reporta la **violación máxima** (valor y posición), no solo pass/fail (transparencia).
- T3 (equivalencia estadística tras 20 épocas) se define en `training.md` §9.
- No se exige igualdad bit a bit **entre** motores; sí dentro de un motor entre ejecuciones (determinismo, `training.md` §8).

## 13. Matriz de motores y hoja de ruta

| Motor | Estado | Notas de preparación |
|---|---|---|
| python | FASE 2+ (siguiente) | Referencia y generador de artefactos |
| c | Futuro | Sin dependencias: TensorBundle, dataset y perm son crudos; el lector de CSV del manifiesto es opcional |
| rust | Futuro | Convivencia con `rust-ai-core` por decidir (decisión P-005) |
| cpp | Futuro | — |
| java | Futuro | uint32 LE se lee como int (valores < 8000, sin bit de signo) |
| go | Futuro | — |

## 14. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción completa (FASE 0). |
| 1.0-draft r1 | 2026-11-11 | FASE 1: forma del índice TensorBundle (ndim ≤ 4), política git coherente con D-017, generador de pesos (D-018) (ver `docs/validation-fase1.md`). |
| 1.0-draft r2 | 2026-11-11 | CIFAR-10 binario (D-028) y flujo de selección (D-029) reflejados. |
