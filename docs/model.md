# model.md — Especificación contractual de la red neuronal

> **Estado**: FASE 0 (especificación). Versión 1.0-draft. Pendiente de aprobación.
> Este documento es **contractual**: las seis implementaciones (Python, C, Rust, C++, Java, Go) deben reproducir exactamente lo aquí definido. Cualquier desviación debe declararse en el informe del motor y contar con una decisión registrada en `docs/decisions.md`.

---

## 1. Identificación del modelo

**CNN-32 Binary Classifier** (clasificador binario de perros).

| Propiedad | Valor |
|---|---|
| Tarea | Clasificación binaria: DOG (1) / NOT_DOG (0) |
| Entrada | Imagen RGB 32 × 32, valores FP32 en [0, 1] |
| Formato interno | NCHW (batch, canales, alto, ancho) |
| Precisión | FP32 (IEEE 754 binary32) en todo el grafo |
| Parámetros | 78.945 |
| Salida | 1 logit → probabilidad vía Sigmoid |

## 2. Convenciones globales

Estas convenciones son vinculantes para todas las implementaciones:

1. **Orden de ejes**: los tensores de imágenes usan NCHW: `[N, C, H, W]`.
2. **Orden de memoria**: row-major (orden C). La última dimensión (`W`) es contigua. Coincide con el formato del fichero `TensorBundle` (ver `docs/architecture.md` §5).
3. **Precisión**: todas las operaciones del grafo (forward, backward, pérdida, actualización) se calculan en FP32. No se permiten conversiones intermedias a FP64 ni FP16 dentro del entrenamiento. Las **pruebas** (gradient check numérico) pueden usar FP64 internamente como herramienta de verificación, pero nunca forman parte del motor.
4. **Endianness**: little-endian en todos los ficheros intercambiables (ver `docs/architecture.md`).
5. **Reducción de la pérdida**: media sobre el batch (ver §8).
6. **Estrategia interna**: el contrato define la **función matemática** de cada operación. Cada motor elige su estrategia de implementación (bucles directos, im2col + GEMM, etc.) y la documenta en su informe (decisión D-001). Los resultados deben cumplir las tolerancias de `docs/architecture.md` §12.
7. **Determinismo**: un motor ejecutado dos veces con los mismos artefactos compartidos debe producir resultados bit-idénticos en modalidad mono-hilo (verificación interna obligatoria de cada motor).

## 3. Arquitectura

```
Entrada (N, 3, 32, 32)
   │
   ▼
[Conv2D  16 filtros, kernel 3×3, stride 1, VALID, bias]   → (N, 16, 30, 30)
[ReLU]                                                    → (N, 16, 30, 30)
[MaxPool 2×2, stride 2 (floor)]                           → (N, 16, 15, 15)
   │
   ▼
[Conv2D  32 filtros, kernel 3×3, stride 1, VALID, bias]   → (N, 32, 13, 13)
[ReLU]                                                    → (N, 32, 13, 13)
[MaxPool 2×2, stride 2 (floor)]                           → (N, 32, 6, 6)
   │
   ▼
[Flatten NCHW]                                            → (N, 1152)
   │
   ▼
[Dense 1152 → 64, bias]                                   → (N, 64)
[ReLU]                                                    → (N, 64)
   │
   ▼
[Dense 64 → 1, bias]                                      → (N, 1)   ← LOGIT
   │
   ▼
[Sigmoid] (solo para probabilidad/inferencia)            → (N, 1)   ∈ (0, 1)
```

**Nota crítica sobre la salida**: la capa final produce un **logit**. La función de pérdida (Binary Cross-Entropy con logits, §8) consume el logit directamente. Sigmoid se aplica solo para obtener probabilidades en inferencia y métricas. No se calcula nunca `log(sigmoid(z))`.

### 3.1 Tabla de dimensiones

| # | Capa | Configuración | Entrada | Salida |
|---|---|---|---|---|
| 0 | Input | RGB 32×32, FP32, [0,1] | — | (N, 3, 32, 32) |
| 1 | Conv2D_1 | 16 filtros, 3×3, stride 1, sin padding (VALID), bias | (N, 3, 32, 32) | (N, 16, 30, 30) |
| 2 | ReLU_1 | elemento a elemento | (N, 16, 30, 30) | (N, 16, 30, 30) |
| 3 | MaxPool_1 | 2×2, stride 2, floor | (N, 16, 30, 30) | (N, 16, 15, 15) |
| 4 | Conv2D_2 | 32 filtros, 3×3, stride 1, VALID, bias | (N, 16, 15, 15) | (N, 32, 13, 13) |
| 5 | ReLU_2 | elemento a elemento | (N, 32, 13, 13) | (N, 32, 13, 13) |
| 6 | MaxPool_2 | 2×2, stride 2, floor | (N, 32, 13, 13) | (N, 32, 6, 6) |
| 7 | Flatten | NCHW → vector | (N, 32, 6, 6) | (N, 1152) |
| 8 | Dense_1 | 1152 → 64, bias | (N, 1152) | (N, 64) |
| 9 | ReLU_3 | elemento a elemento | (N, 64) | (N, 64) |
| 10 | Dense_2 | 64 → 1, bias | (N, 64) | (N, 1) — logit |
| 11 | Sigmoid | solo inferencia/métricas | (N, 1) | (N, 1) — probabilidad |

### 3.2 Verificación de dimensiones

- Conv2D_1: `30 = 32 − 3 + 1` ✓
- MaxPool_1: `15 = 30 / 2` (30 par) ✓
- Conv2D_2: `13 = 15 − 3 + 1` ✓
- MaxPool_2: `6 = floor(13 / 2)` — **13 es impar**: la fila 12 y la columna 12 de la entrada de MaxPool_2 se descartan (ver §4.3) ✓
- Flatten: `1152 = 32 · 6 · 6` ✓

### 3.3 Parámetros por capa

| Capa | Tensor | Forma | Elementos |
|---|---|---|---:|
| Conv2D_1 | `conv1.weight` | (16, 3, 3, 3) | 432 |
| Conv2D_1 | `conv1.bias` | (16,) | 16 |
| Conv2D_2 | `conv2.weight` | (32, 16, 3, 3) | 4.608 |
| Conv2D_2 | `conv2.bias` | (32,) | 32 |
| Dense_1 | `dense1.weight` | (64, 1152) | 73.728 |
| Dense_1 | `dense1.bias` | (64,) | 64 |
| Dense_2 | `dense2.weight` | (1, 64) | 64 |
| Dense_2 | `dense2.bias` | (1,) | 1 |
| **Total** | | | **78.945** |

Los nombres de tensor (`conv1.weight`, …) son **canónicos** y se usan en todos los ficheros intercambiables (pesos iniciales, checkpoints, vectores de prueba).

---

## 4. Especificación de operaciones

Cada operación se define con: forward, backward (gradiente analítico), formas y convenciones. La notación usa índices: `n` = muestra del batch, `c` = canal, `h`/`y` = fila, `w`/`x` = columna, `o` = filtro/neurona de salida.

### 4.1 Conv2D

**Configuración**: stride 1, sin padding (VALID), con bias. Kernel cuadrado `K×K` (K=3 en ambas capas).

**Formas**:
- Entrada: `X ∈ (N, C, H, W)`
- Pesos: `Wm ∈ (O, C, K, K)` (o = filtro de salida, c = canal de entrada, u/v = offset del kernel)
- Bias: `B ∈ (O,)`
- Salida: `Y ∈ (N, O, H_out, W_out)` con `H_out = H − K + 1`, `W_out = W − K + 1`

**Forward** (definición matemática):

```
Y[n, o, i, j] = B[o] + Σ_{c=0}^{C-1} Σ_{u=0}^{K-1} Σ_{v=0}^{K-1}
                     Wm[o, c, u, v] · X[n, c, i+u, j+v]
```

para `i ∈ [0, H_out)`, `j ∈ [0, W_out)`.

**Backward**. Dado el gradiente de la salida `dY ∈ (N, O, H_out, W_out)`:

```
dB[o]        = Σ_{n,i,j} dY[n, o, i, j]

dWm[o,c,u,v] = Σ_{n,i,j} dY[n, o, i, j] · X[n, c, i+u, j+v]

dX[n,c,p,q]  = Σ_{o} Σ_{u,v}  dY[n, o, p−u, q−v] · Wm[o, c, u, v]
```

donde en `dX` solo intervienen los términos con `0 ≤ p−u < H_out` y `0 ≤ q−v < W_out` (comprobación de límites obligatoria).

**Estrategia de implementación**: libre (bucles directos, im2col + GEMM, winograd, …), documentada por motor (D-001). La función matemática y las formas son fijas.

**Optimización permitida en el modelo completo**: la capa Conv2D_1 es la primera del grafo; su `dX` no se necesita. Un motor puede omitir su cálculo, pero la operación aislada debe implementarlo correctamente para pasar los vectores de prueba (ver `docs/architecture.md` §10).

### 4.2 ReLU

**Forward**: `Y = max(X, 0)` elemento a elemento.

**Backward**: `dX = dY ⊙ 𝟙[X > 0]` (producto de Hadamard con la máscara).

**Convención en el punto x = 0**: la derivada es **0** (subgradiente). Decisión D-008. El motor puede materializar la máscara o guardar la entrada; el resultado debe ser idéntico.

### 4.3 MaxPool2D

**Configuración**: ventana 2×2, stride 2, **floor** (sin padding).

**Formas**:
- Entrada: `X ∈ (N, C, H, W)`
- Salida: `Y ∈ (N, C, H_out, W_out)` con `H_out = floor(H/2)`, `W_out = floor(W/2)`

**Forward**:

```
Y[n, c, i, j] = max( X[n,c,2i,  2j],   X[n,c,2i,  2j+1],
                     X[n,c,2i+1,2j],   X[n,c,2i+1,2j+1] )
```

**Comportamiento con dimensiones impares**: las filas/columnas sobrantes se **descartan**. Con H=13: `H_out = 6`, la ventana cubre las filas 0–11 y la fila 12 se ignora. No hay padding ni inclusión parcial.

**Backward**: el gradiente se enruta **íntegro** a la posición del máximo:

```
dX[n,c,2i+r, 2j+s] += dY[n,c,i,j]   sii (r,s) = argmax del bloque
```

Las posiciones de `dX` que no son máximo de ningún bloque (incluidas la fila y columna descartadas con dimensión impar) valen **0**: `dX` se inicializa a cero.

**Desempate (ties)**: si hay varios máximos iguales dentro del bloque, el gradiente va **solo al primero en orden raster** (r primero, s después; es decir, el máximo con menor `2i+r` y, en caso de igualdad, menor `2j+s`). Decisión D-008. Alternativa descartada: reparto uniforme (complica la equivalencia entre motores sin beneficio).

### 4.4 Flatten

Reordena `(N, C, H, W)` → `(N, C·H·W)` respetando el orden NCHW row-major:

```
índice plano k = c · (H·W) + h · W + w
```

**Backward**: reshape inverso de `(N, C·H·W)` → `(N, C, H, W)`. Sin operaciones aritméticas.

### 4.5 Dense

**Formas** (convención de pesos: salida × entrada):
- Entrada: `X ∈ (N, I)`
- Pesos: `Wd ∈ (O, I)`
- Bias: `Bd ∈ (O,)`
- Salida: `Y ∈ (N, O)`

**Forward**:

```
Y[n, o] = Bd[o] + Σ_{i=0}^{I-1} Wd[o, i] · X[n, i]
```

Forma matricial equivalente (permitida): `Y = X · Wdᵀ + Bd`.

**Backward**. Dado `dY ∈ (N, O)`:

```
dWd[o, i] = Σ_{n} dY[n, o] · X[n, i]        (forma matricial: dWd = dYᵀ · X)
dBd[o]    = Σ_{n} dY[n, o]
dX[n, i]  = Σ_{o} dY[n, o] · Wd[o, i]       (forma matricial: dX = dY · Wd)
```

### 4.6 Sigmoid

**Forward**, implementación numéricamente estable obligatoria:

```
σ(z) = 1 / (1 + e^(−z))          si z ≥ 0
σ(z) = e^(z) / (1 + e^(z))       si z < 0
```

(nunca se evalúa `exp` con argumento positivo sin acotar).

**Derivada**: `σ'(z) = σ(z) · (1 − σ(z))`.

**Uso**: en este modelo, Sigmoid aparece **solo** tras el logit, para obtener probabilidades. La pérdida usa logits (§8), por lo que el backward de la pérdida no pasa por Sigmoid (se simplifica analíticamente, ver §8.2).

---

## 5. Forward del modelo completo

Dado un batch `X ∈ (N, 3, 32, 32)`:

1. `A1 = Conv2D_1(X)` → `(N, 16, 30, 30)`
2. `A2 = ReLU(A1)`
3. `A3 = MaxPool_1(A2)` → `(N, 16, 15, 15)`
4. `A4 = Conv2D_2(A3)` → `(N, 32, 13, 13)`
5. `A5 = ReLU(A4)`
6. `A6 = MaxPool_2(A5)` → `(N, 32, 6, 6)`
7. `A7 = Flatten(A6)` → `(N, 1152)`
8. `A8 = Dense_1(A7)` → `(N, 64)`
9. `A9 = ReLU(A8)`
10. `Z = Dense_2(A9)` → `(N, 1)` — **logits**
11. (Solo inferencia/métricas) `P = σ(Z)` → `(N, 1)`

**Entradas que deben conservarse para el backward** (por muestra, batch N=32):

| Tensor | Forma | Necesario para |
|---|---|---|
| X (entrada) | (N,3,32,32) | dW/dB de Conv2D_1 |
| A1 | (N,16,30,30) | ReLU_1 (máscara del backward) |
| A2 | (N,16,30,30) | MaxPool_1 (posiciones de máximo o entrada) |
| A3 | (N,16,15,15) | Conv2D_2 |
| A4 | (N,32,13,13) | ReLU_2 |
| A5 | (N,32,13,13) | MaxPool_2 |
| A6/A7 | (N,32,6,6) = (N,1152) | Dense_1 |
| A8 | (N,64) | ReLU_3, Dense_2 |
| A9 | (N,64) | Dense_2 |

**Memoria de activaciones** (N=32, FP32, conservando todas las entradas listadas): 1.522.176 valores ≈ 5,8 MiB, más los buffers internos de la estrategia elegida (p. ej. im2col), que cada motor documenta. Un motor que guarde máscaras en lugar de duplicar tensores (p. ej. solo `A2` en vez de `A1`+`A2`) reduce esta cifra; es una optimización de memoria legítima y documentable.

---

## 6. Backward del modelo completo

Orden inverso exacto. Sea `dZ ∈ (N, 1)` el gradiente de la pérdida respecto a los logits (definido en §8.2).

1. `Dense_2.backward(dZ)` → produce `dW(dense2)`, `dB(dense2)`, `dA9`
2. `ReLU_3.backward(dA9)` → `dA8`
3. `Dense_1.backward(dA8)` → `dW(dense1)`, `dB(dense1)`, `dA7`
4. `Flatten.backward(dA7)` → `dA6`
5. `MaxPool_2.backward(dA6)` → `dA5`
6. `ReLU_2.backward(dA5)` → `dA4`
7. `Conv2D_2.backward(dA4)` → `dW(conv2)`, `dB(conv2)`, `dA3`
8. `MaxPool_1.backward(dA3)` → `dA2`
9. `ReLU_1.backward(dA2)` → `dA1`
10. `Conv2D_1.backward(dA1)` → `dW(conv1)`, `dB(conv1)`; `dX` opcional (§4.1)

Los gradientes de los parámetros se acumulan **sumando sobre todas las muestras del batch** (la reducción temporal ya la aplica la pérdida vía `dZ`, ver §8.2).

---

## 7. Inicialización de parámetros

**Regla general**: los bias se inicializan a **0**. Las matrices de pesos se muestrean de una normal de media 0:

- **He initialization** (capas seguidas de ReLU): `W ~ N(0, std²)` con
  `std = sqrt(2 / fan_in)`
- **Xavier initialization** (capa de salida, seguida de Sigmoid): `W ~ N(0, std²)` con
  `std = sqrt(2 / (fan_in + fan_out))`

`fan_in` = número de entradas de la capa (canales·kH·kW en conv; `I` en dense). `fan_out` = número de salidas.

| Capa | Init | fan_in | fan_out | std |
|---|---|---:|---:|---|
| conv1 | He | 3·3·3 = 27 | — | sqrt(2/27) ≈ 0,272166 |
| conv2 | He | 16·3·3 = 144 | — | sqrt(2/144) ≈ 0,117851 |
| dense1 | He | 1152 | — | 1/24 ≈ 0,041667 |
| dense2 | Xavier | 64 | 1 | sqrt(2/65) ≈ 0,175412 |

### 7.1 Fuente de verdad de los pesos iniciales

Los pesos iniciales se generan **una sola vez** (herramienta Python, Fase 2) con una semilla maestra registrada y se almacenan en:

```
shared/initial_weights/initial_weights.tb   (formato TensorBundle, docs/architecture.md §5)
```

Todos los motores cargan este fichero. **No se permite** que un motor genere sus propios pesos aleatorios en ejecución normal. La semilla y el procedimiento de regeneración se documentan en `docs/architecture.md` §7 (decisión D-006).

---

## 8. Función de pérdida: Binary Cross-Entropy con logits

### 8.1 Definición

Para un batch de `N` muestras con logits `z_n` y etiquetas binarias `y_n ∈ {0, 1}`:

```
L = (1/N) · Σ_{n=0}^{N-1} ℓ(z_n, y_n)

ℓ(z, y) = max(z, 0) − z·y + log(1 + e^(−|z|))
```

**Propiedades de estabilidad** (obligatorio preservarlas):
- Nunca se evalúa `e^(z)` con z grande positivo (desbordamiento).
- Nunca se calcula `log(σ(z))` ni `log(1−σ(z))` directamente.
- La fórmula es exacta (no es una aproximación): `max(z,0) − z·y + log(1+e^{−|z|}) ≡ −[y·log σ(z) + (1−y)·log(1−σ(z))]`.

**Reducción**: media aritmética sobre el batch (N=32 en entrenamiento). La suma se realiza en orden de índice `n` creciente, en FP32.

### 8.2 Gradiente respecto al logit

```
∂L/∂z_n = (σ(z_n) − y_n) / N
```

Es decir: `dZ = (σ(Z) − Y) / N`. Este vector inicia la cadena de backward (§6). Obsérvese que la derivada de Sigmoid y la de BCE se han combinado analíticamente: **el backward no ejecuta Sigmoid como capa**.

**Verificación**: la pérdida por muestra `ℓ` cumple `ℓ'(z) = σ(z) − y`, por lo que con la reducción por media el gradiente por muestra queda dividido por N.

---

## 9. Precisión numérica y reglas de coma flotante

1. **FP32 en todo el grafo** (forward, backward, pérdida, actualización SGD).
2. **Acumulaciones**: en FP32. El orden de suma interno de cada operación es parte de la estrategia del motor (D-001) y queda cubierto por las tolerancias (§12 de `docs/architecture.md`).
3. **Prohibiciones**:
   - Conversión a FP16/FP64 en cualquier punto del entrenamiento.
   - Compilación con matemática flotante relajada (`-ffast-math` y equivalentes) en las build de RELEASE medidas. Los NaN/Inf no se "limpian" silenciosamente.
   - FMA (fused multiply-add) **sí está permitido**: está cubierto por la tolerancia y NumPy/BLAS lo usa. (Decisión D-015.)
4. **Chequeos de sanidad**: si aparece NaN o Inf en la pérdida, el entrenamiento aborta con error explícito (no se silencia).

---

## 10. Comportamiento esperado de cada capa (criterios de aceptación)

| Capa | Invariante a verificar en pruebas |
|---|---|
| Conv2D | Con pesos y bias a 0 y entrada constante k → salida 0. Con kernel delta (1 en el centro) → copia desplazada. dW/dB/dX contra diferencias finitas. |
| ReLU | Salida = entrada en los positivos, 0 en negativos; máscara del backward correcta en x=0. |
| MaxPool2D | Salida no mayor que el máximo de la entrada; bloque impar descarta fila/columna sobrante; gradiente preserva la suma total del dY. |
| Flatten | Permutación: reordenar y deshacer recupera el tensor original. Índice plano k = c·HW + h·W + w. |
| Dense | Salida con bias 0 y pesos identidad = entrada (si I=O); gradientes contra diferencias finitas. |
| Sigmoid | σ(0)=0,5; σ(30)≈1 sin Inf; σ(−30)≈0 sin NaN; simetría σ(−z)=1−σ(z) dentro de tolerancia. |
| BCE | ℓ(0, y)=log(2)≈0,6931; ℓ(z grande, y=1)→0; ℓ(z grande, y=0)→z (lineal, sin overflow). |

**Gradient check numérico** (obligatorio en cada motor, Fase 2): para cada operación diferenciable se compara el gradiente analítico con diferencias centrales en FP64:

```
g_num = [f(x+h) − f(x−h)] / (2h)      con h = 1e-5, aritmética FP64
error_rel = ||g_num − g_ana||₂ / max(||g_num||₂, ||g_ana||₂, 1e-12)
criterio: error_rel < 1e-6  para dense/sigmoid/BCE/flatten
          error_rel < 1e-5  para conv/maxpool (más términos de suma)
```

El gradient check usa FP64 **solo como herramienta de prueba**; el motor que se mide sigue siendo FP32 (§2.3).

**Cruces de «kink»**: ReLU y MaxPool no son diferenciables en los empates/ceros; si una perturbación `±h` cruza uno, la diferencia finita falla sin que haya error de implementación. Por eso los casos unitarios usan entradas con separación > 2h y el gradcheck del modelo completo usa lotes pequeños; si un tensor falla con h = 1e-5, se **repiten los mismos elementos con h = 1e-7** (el cruce deja de ocurrir) y se exige la misma tolerancia; se documenta el cruce, nunca se relaja la tolerancia. Un error real falla con ambos pasos. Observado en la FASE 3: un cruce de argmax de MaxPool en `conv1.weight` (1 de 5 semillas).

---

## 11. Vectores de prueba (golden vectors)

Cada operación y el modelo completo tendrán casos deterministas con entrada, parámetros, salida esperada y gradientes esperados, almacenados en `shared/test_vectors/` (estructura y formato en `docs/architecture.md` §10). Son la herramienta de equivalencia entre los seis motores.

---

## 12. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0-draft | 2026-11-10 | Primera redacción completa (FASE 0). |
| 1.0-draft r1 | 2026-11-11 | FASE 1: A1 sin dX de Conv2D_1, dX=0 fuera de máximos, nota sobre kinks en gradcheck (ver `docs/validation-fase1.md`). |
| 1.0-draft r2 | 2026-11-11 | §10: protocolo de repetición con h menor ante cruces de kink. |
