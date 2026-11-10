# dataset.md — Especificación contractual del conjunto de datos

> **Estado**: FASE 0 (especificación). Versión 1.0-draft. Pendiente de aprobación.
> Documento **contractual** para las seis implementaciones. El dataset derivado y sus manifiestos son artefactos compartidos: ningún motor genera sus propios datos.

---

## 1. Propósito y fuente

**Dataset base**: CIFAR-10 ( versión «python »), fuente oficial:

```
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

- 60.000 imágenes RGB 32×32 (50.000 entrenamiento + 10.000 test), 10 clases equilibradas.
- No se aplica *data augmentation* en la primera versión del proyecto.
- El problema se reduce a **binario**: DOG = 1, NOT_DOG = 0.

## 2. Descarga y verificación de integridad

1. Descargar `cifar-10-python.tar.gz` a `shared/dataset/raw/`.
2. Verificar integridad:
   - **MD5 oficial** (publicado en la página de la fuente): `c58f30108f718f92721af3b95e74349a`.
   - **SHA-256**: se fijará y registrará en `shared/dataset/CHECKSUMS.sha256` en la primera descarga real (FASE 4). *(No se inventa el valor ahora: el hash exacto se marca como «pendiente de fijar en Fase 4».)*
3. La descarga la realiza únicamente el script Python `scripts/download_cifar10.py`. Los otros motores **nunca descargan ni leen el archivo original**.

## 3. Interpretación del formato original

Versión «python» de CIFAR-10: archivo tar.gz con:

```
cifar-10-batches-py/
├── data_batch_1 … data_batch_5   (5 × 10.000 imágenes de entrenamiento)
├── test_batch                    (10.000 imágenes de test)
└── batches.meta                  (nombres de clases)
```

Cada `data_batch_N` es un pickle Python con claves `b'data'` y `b'labels'`:

- `data`: array uint8 de forma (10000, 3072). **Por imagen**, los 3072 bytes se organizan en 3 planos consecutivos de 1024 bytes:
  - bytes 0–1023: plano **R** (canal rojo),
  - bytes 1024–2047: plano **G** (canal verde),
  - bytes 2048–3071: plano **B** (canal azul),
  - cada plano es 32×32 en **orden raster** (fila a fila, de arriba a abajo, de izquierda a derecha).
- `labels`: lista de 10.000 enteros 0–9.

Es decir, el formato original ya es planar por canal: interpretar una fila como `(3, 32, 32)` es un *reshape* puro, sin reordenar bytes.

**Clases CIFAR-10** (orden oficial de `batches.meta`):

| id | clase | id | clase |
|---|---|---|---|
| 0 | airplane | 5 | **dog** |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

**Mapeo binario**: `dog (5) → 1`; el resto `→ 0`.

**Importante para la independencia entre lenguajes**: el pickle solo lo lee la herramienta Python una vez. El resto de motores consumen exclusivamente los artefactos derivados de `shared/dataset/` (formato binario propio, §6), por lo que ninguno necesita soporte de pickle.

## 4. Distribución objetivo

| Conjunto | Perros | No perros | Total | Origen |
|---|---:|---:|---:|---|
| Train | 4.000 | 4.000 | 8.000 | pool de entrenamiento (50.000) |
| Validation | 500 | 500 | 1.000 | pool de entrenamiento (50.000) |
| Test | 1.000 | 1.000 | 2.000 | pool de test (10.000) |

- **Validation sale del pool de entrenamiento**, nunca del pool de test. El pool de test solo aporta el conjunto Test.
- Las cuotas por clase negativa (9 clases) se reparten de forma estratificada. El resto de la división se asigna en **orden ascendente de id de clase**:

**Cuotas de clases negativas**

| Clase negativa | Train | Validation | Test |
|---|---:|---:|---:|
| 0 airplane | 445 | 56 | 112 |
| 1 automobile | 445 | 56 | 111 |
| 2 bird | 445 | 56 | 111 |
| 3 cat | 445 | 56 | 111 |
| 4 deer | 445 | 56 | 111 |
| 5 — (dog, positiva) | 4.000 | 500 | 1.000 |
| 6 frog | 444 | 55 | 111 |
| 7 horse | 444 | 55 | 111 |
| 8 ship | 444 | 55 | 111 |
| 9 truck | 444 | 55 | 111 |
| **Total** | **8.000** | **1.000** | **2.000** |

Verificación aritmética: train 4×445+5×444 = 1.780+2.220 = 4.000 ✓; val 5×56+4×55 = 280+220 = 500 ✓; test 112+8×111 = 112+888 = 1.000 ✓. Disponibilidad: cada clase tiene 5.000 ejemplos en el pool de entrenamiento (máx. necesario: 445+56 = 501) y 1.000 en el pool de test (máx. necesario: 112) ✓. El conjunto Test emplea los 1.000 perros del pool de test.

## 5. Selección determinista estratificada

Decisión aprobada (D-003): **permutación canónica pregenerada** + cuotas por clase. El procedimiento lo ejecuta una única vez la herramienta Python (`scripts/prepare_dataset.py`); los motores cargan los resultados.

### 5.1 Algoritmo

1. Construir la lista de **índices globales** del pool de entrenamiento: `data_batch_1[0..9999] … data_batch_5[0..9999]` (global 0–49.999), y del pool de test: `test_batch[0..9999]`.
2. Agrupar por clase CIFAR-10 en **orden ascendente de índice global**.
3. Para cada clase, barajar su lista con **Fisher-Yates** alimentado por el PRNG **splitmix64** (semillas en §5.2).
4. Tomar por clase, en el orden barajado: primero los `n_train(c)` de train, después los `n_val(c)` de validation (pool de entrenamiento); en el pool de test, los primeros `n_test(c)`.

El resultado se vuelca a los manifiestos (§7). Los manifiestos, no este algoritmo, son el artefacto canónico que consumen los motores: el algoritmo existe para auditar y regenerar.

### 5.2 PRNG splitmix64 (referencia)

Estado de 64 bits `x` (sin signo, módulo 2⁶⁴ implícito):

```
siguiente():
    x ← x + 0x9E3779B97F4A7C15
    z ← x
    z ← (z XOR (z >> 30)) · 0xBF58476D1CE4E5B9
    z ← (z XOR (z >> 27)) · 0x94D049BB133111EB
    z ← z XOR (z >> 31)
    devolver z
```

Fisher-Yates (sin sesgo):

```
para i desde n−1 hasta 1:
    j ← siguiente() MOD (i+1)
    intercambiar lista[i] y lista[j]
```

**Semillas** (documentadas y fijas):

| Uso | Semilla inicial `x₀` |
|---|---|
| Selección estratificada | `20261110` |
| Permutación de época `e` (e = 1…20) | `20261110 + e` |

Las permutaciones de entrenamiento (§8) también las genera la herramienta y se almacenan como artefactos compartidos.

### 5.3 Por qué un PRNG propio y no `numpy.random`

La regeneración no debe depender de la versión de NumPy ni de ningún runtime: splitmix64 son 10 líneas en cualquier lenguaje y su flujo es idéntico en los seis. La herramienta Python lo implementa explícitamente (no delega en `random`/`numpy.random`).

## 6. Formato del dataset derivado (artefacto compartido)

Ruta: `shared/dataset/`. Archivos binarios **little-endian**, sin cabecera:

| Fichero | Contenido | Tamaño |
|---|---|---|
| `train_images.bin` | 8.000 × 3072 uint8 | 24.576.000 B ≈ 24,6 MB |
| `train_labels.bin` | 8.000 uint8 (0/1) | 8.000 B |
| `val_images.bin` | 1.000 × 3072 uint8 | 3.072.000 B ≈ 3,1 MB |
| `val_labels.bin` | 1.000 uint8 | 1.000 B |
| `test_images.bin` | 2.000 × 3072 uint8 | 6.144.000 B ≈ 6,1 MB |
| `test_labels.bin` | 2.000 uint8 | 2.000 B |

**Layout de cada imagen** (3072 bytes): tres planos consecutivos R, G, B; cada plano 32×32 en orden raster. Equivalente a un tensor `(3, 32, 32)` en row-major: el byte `k`-ésimo de la imagen es `plano k div 1024, fila (k mod 1024) div 32, columna (k mod 1024) mod 32`.

Es el mismo orden del formato original: la conversión es copia + reshape, sin reordenar.

**Orden de las muestras**: la posición `p` (0-based) dentro de cada fichero corresponde a la fila `p` del manifiesto correspondiente. Las imágenes de un batch de entrenamiento se leen en posiciones consecutivas según la permutación de época (§8).

**Etiquetas binarias**: uint8 con valor 0 (NOT_DOG) o 1 (DOG). La clase CIFAR original queda registrada en el manifiesto.

## 7. Manifiestos de índices

Ruta: `shared/dataset/manifests/`. CSV con cabecera, UTF-8, separador `,`:

```csv
position,source,source_index,cifar_class,binary_label
0,data_batch_1,4964,5,1
1,data_batch_3,1099,8,0
...
```

| Columna | Significado |
|---|---|
| `position` | Posición 0-based en `*_images.bin` / `*_labels.bin` |
| `source` | Lote de origen (`data_batch_1`…`data_batch_5`, `test_batch`) |
| `source_index` | Índice 0-based dentro de ese lote original |
| `cifar_class` | Clase original CIFAR-10 (0–9) |
| `binary_label` | Etiqueta binaria (0/1) |

Un fichero por conjunto: `train.csv`, `val.csv`, `test.csv`.

## 8. Permutaciones de entrenamiento (artefacto compartido)

Ruta: `shared/permutations/`. Un fichero por época:

```
train_epoch_001.perm … train_epoch_020.perm
```

- Formato: 8.000 uint32 little-endian, valores ∈ [0, 8000), permutación completa de las posiciones de train (sin repetición).
- Generadas con Fisher-Yates + splitmix64, semilla `20261110 + e` (§5.2).
- Validación: cada época declara cuota idéntica de perros/no perros (el barajado no cambia la composición) y es permutación exacta (suma de índices = Σ 0..7999 y sin duplicados).
- Los motores **cargan** estas permutaciones; no las generan. Así el orden de presentación de muestras es idéntico en los seis lenguajes sin depender de ningún PRNG en runtime.
- Validation y test se evalúan **en orden de manifiesto**, sin barajar.

## 9. Normalización y preprocesamiento (idéntico en los seis motores)

El único preprocesamiento es:

```
x_fp32 = (uint8_valor / 255.0)   →  rango [0, 1]
```

- Conversión uint8 → FP32 exacta (todos los valores 0–255 son representables en FP32).
- División por la constante 255.0: operación IEEE 754 correctamente redondeada → **bit-idéntica en los seis lenguajes**.
- NO se resta media ni se divide por desviación típica. NO hay data augmentation. NO hay reescalado (las imágenes ya son 32×32).
- La conversión la realiza **cada motor al cargar** (no se almacena FP32 en disco): se ahorra 4× de espacio y la operación es bit-exacta.

**Orden de canales**: se conserva el original RGB (canal 0 = R, 1 = G, 2 = B) en todo el sistema.

## 10. Validación de ausencia de solapamientos

`scripts/prepare_dataset.py` verifica y aborta con error si falla:

1. `train ∩ val = ∅` sobre pares `(source, source_index)` del pool de entrenamiento.
2. `train ∪ val` no contiene pares duplicados internamente; ídem `test`.
3. El pool de test solo alimenta a `test` (por construcción, los orígenes difieren; se comprueba igualmente).
4. Cuotas exactas por clase y por conjunto (tabla §4).
5. Suma de tamaños: 8.000 / 1.000 / 2.000 y balance 50/50 exacto en los tres conjuntos.
6. SHA-256 de cada artefacto escrito en `shared/dataset/CHECKSUMS.sha256`, verificable por cualquier motor antes de entrenar.

## 11. Política de almacenamiento y git

- **No se sube a git** ningún binario de datos (`shared/dataset/*.bin`, `*.perm` y el tar.gz original quedan fuera mediante `.gitignore`).
- **Sí se suben**: manifiestos CSV, `CHECKSUMS.sha256`, `.gitignore` y los scripts que regeneran todo.
- Regeneración: `scripts/download_cifar10.py` → `scripts/prepare_dataset.py` reproduce bit a bit los mismos artefactos (PRNG propio + semillas fijas), verificable contra los SHA-256 de los manifiestos.
- Los pesos de los datasets derivados se documentan para dimensionar memoria: ~33,8 MB en disco; en RAM (FP32) train = 8.000 × 3 × 32 × 32 × 4 B ≈ 94 MB si se carga entero, o por lotes si el motor lo prefiere (elección libre de cada motor, documentada).

## 12. Qué consume cada motor (resumen)

Un motor ajeno a Python necesita únicamente:

```
shared/dataset/train_images.bin + train_labels.bin
shared/dataset/val_images.bin  + val_labels.bin
shared/dataset/test_images.bin + test_labels.bin
shared/dataset/manifests/*.csv        (auditoría/debug, opcional en runtime)
shared/permutations/train_epoch_*.perm
shared/dataset/CHECKSUMS.sha256       (verificación opcional)
```

No necesita: el tar.gz original, pickle, la fuente HTTP ni el PRNG.

## 13. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción completa (FASE 0). |
