# training.md — Especificación contractual del entrenamiento

> **Estado**: FASE 0 (especificación). Versión 1.0-draft. Pendiente de aprobación.
> Documento **contractual**. Define el procedimiento exacto de entrenamiento que los seis motores ejecutan. La matemática de capas y pérdida vive en `docs/model.md`; aquí se define el ciclo, la actualización de parámetros y la reproducibilidad.

---

## 1. Configuración (hiperparámetros fijos)

| Parámetro | Valor |
|---|---|
| Epochs | 20 |
| Batch size | 32 (exacto; 8000/32 = 250 lotes por época, sin lotes parciales) |
| Optimizer | SGD |
| Learning rate | 0,01 (constante, sin decaimiento ni plan) |
| Momentum | 0 |
| Weight decay | 0 |
| Loss | Binary Cross-Entropy con logits (media sobre el lote) |
| Precisión | FP32 |
| Early stopping | Desactivado |
| Selección de modelo | Ninguna: el modelo final es el de la época 20 |
| Gradient clipping | Ninguno |
| Shuffle | Determinista, por permutación compartida (`shared/permutations/`) |
| Data augmentation | Ninguna |
| Umbral de decisión | logit ≥ 0 ⟺ DOG (equivale a σ ≥ 0,5) |

La actualización SGD queda reducida, con momentum y weight decay a cero, a:

```
w ← w − 0,01 · g
```

aplicada a **todos los parámetros entrenables** (pesos y bias) tras cada lote. No hay estado de optimizador que guardar.

## 2. Artefactos de entrada (todos compartidos, ninguno generado en runtime)

| Artefacto | Ruta | Uso |
|---|---|---|
| Pesos iniciales | `shared/initial_weights/initial_weights.tb` | Estado inicial idéntico en los seis motores |
| Permutaciones de época | `shared/permutations/train_epoch_001.perm` … `train_epoch_020.perm` | Orden de presentación |
| Imágenes/etiquetas | `shared/dataset/{train,val,test}_{images,labels}.bin` | Datos |

**Consecuencia clave de diseño**: el entrenamiento no usa ningún generador aleatorio en runtime. Dados los artefactos, dos ejecuciones del mismo motor en modo mono-hilo deben producir **resultados bit-idénticos** (verificación obligatoria, §8.2).

## 3. Ciclo de entrenamiento (algoritmo exacto)

```
cargar initial_weights.tb                       # 8 tensores canónicos
para e en 1 … 20:
    perm ← cargar shared/permutations/train_epoch_{e:03d}.perm
    para b en 0 … 249:                          # 250 lotes
        idx ← perm[b·32 … (b+1)·32)             # 32 posiciones
        X ← float32(train_images[idx]) / 255.0  # (32, 3, 32, 32)
        Y ← train_labels[idx]                   # (32,) ∈ {0,1}
        Z ← forward(X)                          # logits (32, 1), docs/model.md §5
        L ← BCE_con_logits(Z, Y)               # escalar, docs/model.md §8
        dZ ← (σ(Z) − Y) / 32
        g ← backward(dZ)                       # gradientes de parámetros, docs/model.md §6
        w ← w − 0,01 · g                        # SGD puro
        registrar L (y tiempo del lote)
    (val_loss, val_acc) ← validar()             # §4
    guardar checkpoint época e                  # §6
    escribir resumen de época en el registro    # §5
```

Notas:

- La normalización `/255.0` se aplica en FP32 al cargar cada lote; nunca se precalcula en disco (decisión `docs/dataset.md` §9).
- Los gradientes de parámetros son la suma sobre las 32 muestras del lote (la división por N ya está dentro de `dZ`; ver `docs/model.md` §6 y §8.2).
- Todos los parámetros se actualizan tras **cada** lote (sin acumulación entre lotes).
- Si `L` es NaN o Inf: abortar con error explícito (no continuar, no silenciar).

## 4. Validación

Se ejecuta **tras cada época** (20 validaciones por entrenamiento):

```
suma_ℓ ← 0;  correctos ← 0
para cada lote de 32 muestras de val (31 lotes de 32 + 1 lote de 8):
    Z ← forward(X)                              # sin backward
    suma_ℓ ← suma_ℓ + Σ_n ℓ(z_n, y_n)          # suma de pérdidas por muestra
    correctos ← correctos + Σ_n [ (z_n ≥ 0) == (y_n == 1) ]   # decisión por logit
val_loss ← suma_ℓ / 1000                        # media sobre TODO el conjunto
val_acc  ← correctos / 1000
```

- La pérdida de validación es la **media sobre el conjunto completo** (1000 muestras), no la media de medias de lote: el último lote tiene 8 muestras y ponderaría igual que uno de 32 si se promediara por lotes. La suma acumulada se hace en orden de manifiesto, en FP32.
- La decisión de clasificación usa el **logit** (≥ 0 ⟺ DOG). Es equivalente a σ ≥ 0,5 y evita calcular Sigmoid para las métricas de conteo; la fórmula estable de BCE (`docs/model.md` §8) tampoco necesita σ: opera directamente sobre el logit.
- Test se evalúa igual (62 lotes de 32 + 1 de 16, media sobre 2000) pero **solo al final del entrenamiento**, no por época.

## 5. Registro de métricas

Salida obligatoria por motor: `train_log.jsonl` (JSON Lines UTF-8, una línea por época):

```json
{"engine": "python", "run": "reference_001", "epoch": 1, "train_loss": 0.6831, "val_loss": 0.6512, "val_accuracy": 0.6230, "epoch_time_s": 12.34}
```

- `train_loss` = media de las pérdidas de los 250 lotes de la época (todos los lotes pesan 32 → media de medias de lote = media por muestra).
- `epoch_time_s` = tiempo de cómputo de la época (bucles de entrenamiento), sin incluir validación ni I/O de checkpoint. El desglose de tiempos completo se define en `docs/benchmark.md`.
- Adicional (opcional): `batch_log.csv` con `epoch,batch,loss,batch_time_s` para análisis por lote.
- Tras el entrenamiento: escribir `final_metrics.json` con `val_loss`, `val_accuracy`, `test_loss`, `test_accuracy`, y los logits de test (§7).

## 6. Checkpoints

- Al final de **cada época** se escribe `checkpoint_epoch_{e:03d}.tb` (formato TensorBundle, `docs/architecture.md` §5) con: los 8 tensores de parámetros + los tensores escalares `meta.*` de `architecture.md` §5.5 (`epoch`, `train_loss`, `val_loss`, `val_accuracy`, `contract_version`).
- Además se copia a `checkpoint_latest.tb` (sobrescrito cada época).
- Al terminar la época 20: `model_final.tb` (idéntico al checkpoint de la época 20).
- Tamaño por checkpoint: 78.945 × 4 B ≈ 308 KB; 20 épocas ≈ 6,3 MB.
- **No se selecciona el mejor modelo**: early stopping desactivado y sin criterio de selección. El modelo evaluado en test es siempre el de la época 20. Documentar esta decisión evita comparar motores con criterios de selección distintos.

## 7. Guardado y carga del modelo

- **Guardar**: los 8 tensores canónicos (`conv1.weight`, `conv1.bias`, `conv2.weight`, `conv2.bias`, `dense1.weight`, `dense1.bias`, `dense2.weight`, `dense2.bias`) en TensorBundle (en checkpoints, más los tensores `meta.*`; no hay JSON embebido, `architecture.md` §5.5).
- **Cargar**: un motor debe poder (a) reanudar conceptualmente desde cualquier checkpoint (el formato lo permite) y (b) cargar `model_final.tb` para inferencia sin ejecutar entrenamiento.
- **Logits de test**: al final del entrenamiento, cada motor escribe `test_logits.csv` con cabecera `position,logit` (2000 filas, orden de manifiesto, formato `%.9g`: 9 cifras significativas, ida y vuelta exacta de un FP32). Las métricas de calidad completas (precision/recall/F1/AUC/matriz de confusión) las calcula un evaluador común (FASE 7) a partir de este fichero y `test_labels.bin`, garantizando que el cálculo de métricas es idéntico para todos los motores y no forma parte del código medido.

## 8. Reproducibilidad

1. **Sin azar en runtime**: pesos iniciales, permutaciones y datos provienen de artefactos compartidos. No hay semillas en el motor de entrenamiento.
2. **Determinismo interno**: dos ejecuciones del mismo motor, misma máquina, modo mono-hilo, deben producir checkpoints bit-idénticos. Verificación: comparar SHA-256 de `model_final.tb` entre dos ejecuciones (prueba obligatoria de cada motor, FASE 5).
3. **Independencia del entorno**: las bibliotecas matemáticas (BLAS, libm) pueden variar entre sistemas; el contrato exige FP32 sin fast-math y fija las tolerancias de §10 para la equivalencia entre motores. El determinismo bit a bit solo se exige *dentro* de un mismo motor ejecutado en la misma máquina y modo.
4. **Auditoría**: los artefactos compartidos se verifican con `shared/dataset/CHECKSUMS.sha256` antes de entrenar (opcional pero recomendado por motor).

## 9. Control de divergencia entre motores

El entrenamiento es un proceso caótico: diferencias de redondeo del orden de 1e-7 en una convolución pueden amplificarse a lo largo de 20 épocas. La estrategia es **por capas**, de lo exacto a lo estadístico:

| Nivel | Qué se compara | Criterio |
|---|---|---|
| T1 — Operación | Salidas y gradientes de cada operación sobre vectores dorados (`shared/test_vectors/`) | Tolerancias de `docs/architecture.md` §12 (1e-5 en salidas, 1e-4 en gradientes) |
| T2 — Un paso de entrenamiento | Tras el lote 1 de la época 1 (lote dorado compartido): pérdida, gradientes y pesos resultantes contra los valores de referencia | rel 1e-4 |
| T3 — Entrenamiento completo | `val_accuracy` final dentro de ±0,03 y `val_loss` final dentro de ±10 % respecto a la referencia Python | Banda provisional (FASE 7) |

- T1 y T2 son **obligatorios** para declarar un motor matemáticamente equivalente; si fallan, localizan la capa o el orden de operaciones divergente.
- T3 es una banda estadística: si un motor la supera, el protocolo exige investigar (revisar T1/T2, orden de acumulación, FMA) **antes** de cuestionar la banda. El objetivo es entender la divergencia, no esconderla.
- La banda T3 se validará y ajustará con los dos primeros motores reales (Python + el segundo que llegue) y quedará fijada por escrito en este §9.

## 10. Criterios de aceptación de la FASE 5

1. El entrenamiento completo (20 épocas) termina sin NaN/Inf y sin errores.
2. `train_loss` global de la época 20 es menor que el de la época 1 (convergencia demostrable, sin valor objetivo inventado: los valores reales se registrarán como línea base).
3. Dos ejecuciones del mismo motor producen `model_final.tb` con el mismo SHA-256.
4. Los 20 checkpoints, `train_log.jsonl` y `test_logits.csv` existen y son legibles por el evaluador.
5. Los pesos iniciales cargados son bit-idénticos a `shared/initial_weights/initial_weights.tb` (verificación de carga).
6. Los casos T2 (paso de entrenamiento dorado) pasan las tolerancias.

## 11. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción completa (FASE 0). |
| 1.0-draft r1 | 2026-11-11 | FASE 1: notación de la validación, σ innecesaria en BCE, metadatos sin JSON, tolerancia T1, formato de logits, erratas (ver `docs/validation-fase1.md`). |
