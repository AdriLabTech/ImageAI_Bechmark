# validation-fase1.md — Validación documental cruzada (FASE 1)

> 2026-11-11. Se leyeron los siete documentos completos, se contrastaron entre sí y se ejecutó `tests/check_spec.py` (aritmética de la especificación, solo biblioteca estándar). Nada ejecutable del motor existe todavía: esto valida la **especificación**, no una implementación.

## 1. Resultado de las comprobaciones

`python3 -I tests/check_spec.py` → todas las comprobaciones aritméticas pasan salvo dos marcadas `KNOWN` (defecto P-013, pendiente de decisión). Cubren: dimensiones 30/15/13/6, 1152, 78.945 parámetros, 1.522.176 activaciones, std de inicialización, cuotas por clase contra las tablas de `dataset.md`, lotes (250 / 31+8 / 62+16), tamaños de fichero y de TensorBundle, y la regla de atípicos.

Se verificó a mano lo que el script no cubre: fórmulas forward/backward de Conv2D, Dense, ReLU, MaxPool, BCE (`ℓ' = σ(z) − y`, `dZ = (σ − y)/N`), fan_in/fan_out de He/Xavier y la coherencia de `training.md` §3 con `model.md` §5–6. **Las derivadas son correctas.** Lo que verifica la derivación de verdad es el gradcheck de la FASE 2.

## 2. Defectos corregidos (mecánicos; sin cambio de diseño)

| # | Documento | Defecto | Corrección |
|---|---|---|---|
| 1 | `dataset.md` §4 | Tabla de cuotas: *deer* train = 445 → la tabla sumaba 4.001, no 4.000 (la línea de verificación de debajo era correcta y la contradecía) | deer = 444; regla del resto explícita (clases 0–3 / 0–4 / 0) |
| 2 | `architecture.md` §5.2 | `ndim ∈ [1,8]` pero `shape` ocupa 16 B = 4 × uint32: un tensor de 5+ dimensiones no cabría | `ndim ∈ [1,4]` (el máximo real es 4) |
| 3 | `training.md` §6–7 vs `architecture.md` §5.5 | training decía «metadatos JSON embebidos» y `engine` en checkpoints; TensorBundle no tiene JSON ni strings | Metadatos = tensores `meta.*`; sin `engine` |
| 4 | `training.md` §4 | Notación ambigua `[z ≥ 0 == y == 1]` | `[(z ≥ 0) == (y == 1)]` |
| 5 | `training.md` §4 | «la pérdida usa σ internamente»: la fórmula estable de BCE no usa σ | Corregido |
| 6 | `training.md` §9 | T1 «rel 1e-5» vs `architecture.md` §12 (gradientes 1e-4) | Alineado con §12 |
| 7 | `training.md` §7 | Logits con «9 decimales»: pierde cifras en logits pequeños | `%.9g` (ida y vuelta exacta de FP32) |
| 8 | `benchmark.md` §7 | Remitía a «§8 de training» para T3 (es §9) y training decía que la banda estaba en benchmark | Referencias arregladas |
| 9 | `benchmark.md` §8 | `model_size_mb` de ejemplo 0,3158 (solo parámetros) | 0,3167 (316.748 B con 13 tensores) |
| 10 | `architecture.md` §2 | «los binarios no se suben a git» contradecía D-017 (pesos y vectores sí) | Reescrito |
| 11 | `model.md` §5 | A1 «necesario para dX de Conv2D_1» (dX no usa activaciones) | Solo máscara ReLU |
| 12 | `model.md` §4.3 | Faltaba decir que `dX` fuera de máximos (y fila/columna descartada) vale 0 | Añadido |
| 13 | `dataset.md` §5.2 | Fisher-Yates «sin sesgo» (el módulo tiene un sesgo < 2⁻⁵⁰) | Matizado |
| 14 | `training.md` | 6,2 MB → 6,3 MB; erratas | Corregido |
| 15 | `architecture.md` §7 | Pesos iniciales sin generador definido | §7.1 con Box-Muller exacto (D-018) |

Adiciones preventivas: nota sobre cruces de *kink* en el gradcheck (`model.md` §10) y sobre el aplanado de ventana de MaxPool en NumPy (`python.md` §4).

## 3. Decisiones de diseño que **no** se han aplicado (esperan tu aprobación)

P-011 formato binario de CIFAR (tu especificación lo pedía), P-012 flujo del PRNG de selección, P-013 regla de atípicos (la actual no puede activarse con n = 5), P-014 semántica del calentamiento, P-015 carga de datos fuera del cronómetro. Detalle y recomendaciones en `decisions.md` §B.

## 4. Riesgos que quedan (no son defectos de la especificación)

- **Gradcheck del modelo completo** con ReLU/MaxPool: puede fallar por cruces de kink aun con código correcto (ver `model.md` §10).
- **Banda T3 provisional** (D-020): no hay datos reales todavía.
- El SHA-256 del tar.gz de CIFAR se fijará en la FASE 4; el MD5 oficial se leyó de la página (extracto), no se ha descargado nada.
- «Dog vs. resto» con una CNN pequeña y SGD simple dará una exactitud modesta; no hay valor objetivo y no se inventa ninguno.

## 5. Criterio de aceptación de la FASE 1

Dimensiones coinciden ✓ · derivadas correctas ✓ · convenciones consistentes entre documentos ✓ (tras las correcciones) · entrenamiento no contradice al modelo ✓ · dataset compatible ✓ · benchmark mide lo que pretende: **con reservas** (P-013, P-014, P-015). Ninguna reserva bloquea la FASE 2.
