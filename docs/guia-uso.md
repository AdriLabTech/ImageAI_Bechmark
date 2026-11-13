# Guía de uso: pruebas, medición oficial y fotos reales

> Todos los comandos se lanzan desde la carpeta del repositorio: `cd ~/Proyectos/ImageAI-Benchmark`.
> Estado: motor Python (FASES 0-6 hechas; FASE 7 con la medición oficial preparada, **aún sin ejecutar**).

## 0. Qué necesitas tener ya

| Cosa | Comprobación rápida |
|---|---|
| Entorno del motor | `ls implementations/python/.venv/bin/python` |
| Fotos del dataset preparadas | `ls shared/dataset/*.bin shared/permutations | head` |
| Modelo entrenado | `ls implementations/python/runs/reference_001/model_final.tb` |

Si falta algo:
```bash
cd implementations/python && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ../..   # entorno
python3 scripts/download_cifar10.py && python -I scripts/prepare_dataset.py                                   # fotos (descarga ~170 MB, lenta)
cd implementations/python && .venv/bin/python -m src.main train --run reference_001 && cd ../..             # entrenar (~1,5 min)
```

## 1. Pruebas (comprobar que todo funciona)

| Qué comprueba | Comando | Resultado esperado |
|---|---|---|
| Las pruebas del motor (131) | `cd implementations/python && .venv/bin/pytest -q` | `131 passed` (~30 s) |
| Las pruebas del contrato, del evaluador y del cronómetro | `implementations/python/.venv/bin/pytest -q tests` | `25 passed, 1 skipped` (la omitida es la de fotos, ver la fila siguiente) |
| Las herramientas de fotos (necesitan Pillow, que trae el Python del sistema) | `PYTHONPATH=$(ls -d implementations/python/.venv/lib/python3*/site-packages) python3 -m pytest -q tests/test_herramientas.py -p no:cacheprovider` | `6 passed` |
| Prueba de aceptación del propio motor (T1, T2, entrenamiento, inferencia) | `cd implementations/python && .venv/bin/python -m src.main tests` | `57 passed`, código de salida 0 |

Si algo falla, **no sigas**: copia el mensaje y díselo a Claude.

## 2. Medición oficial (rendimiento)

Qué hace: ejecuta 5 veces el entrenamiento completo, mide tiempo, memoria y acierto, comprueba que el resultado es idéntico en las 5, y guarda la **referencia de Python** para comparar los otros lenguajes.

Antes de lanzarla:
1. Enchufa el portátil a la corriente.
2. Cierra navegador, vídeos, compilaciones y todo lo que consuma CPU. El script se niega a empezar si la carga es ≥ 1,0 (`uptime` la muestra).
3. No toques el equipo durante ~9 minutos (usa un solo núcleo y cualquier otra tarea falsea los tiempos).

```bash
bash scripts/medicion_oficial_python.sh
```
Te pregunta `¿Empezar? [s/N]`. Al terminar imprime un resumen. Se guardan:

| Fichero | Contenido |
|---|---|
| `benchmark/results/python/summary.json` | Resumen: rendimiento, recursos, calidad, validez |
| `benchmark/results/python/perf_run_00N.json` | Tiempos detallados de cada ejecución (tiempos por lote, por época...) |
| `benchmark/results/python/resources_run_00N.json` | Memoria y CPU muestreadas cada 0,1 s |
| `benchmark/results/python/session.json` | Condiciones de la sesión: máquina, carga, temperatura, núcleo |
| `benchmark/results/python/sesion.log` | Lo que salió por pantalla |
| `reports/reference/python_reference*` | La referencia de Python (modelo, logits, métricas, hashes) |

Cómo interpretarlo (lo que debe cumplirse para que valga):
- `validity.deterministic: true`: las 5 ejecuciones dieron exactamente el mismo modelo.
- `needs_repeat_rel_std_gt_5pct: false`: los 5 tiempos varían menos del 5 %. Si sale `true`, hay que repetir la sesión con el equipo más tranquilo.
- `any_outlier: false`: ninguna ejecución se desvió de forma anómala. Si hay alguna, no se borra: se anota y se investiga.

**Límites que debes conocer** (se documentarán en el informe):
- El portátil se calienta durante la sesión (en una prueba corta subió de 59 a 78 °C); `session.json` lo registra. Si las últimas ejecuciones salen más lentas, es una causa posible.
- El procesador mezcla núcleos rápidos y de ahorro; el arnés fija el núcleo 0 (núcleo rápido) por contrato.
- Los MB son MiB (1 MB = 1.048.576 bytes); por eso el modelo sale como 0,3021 MB y no 0,3167.
- Las horas de desarrollo y las notas de dificultad quedan como `null`: las aportas tú.

Para repetirla, mueve o borra tú a mano `benchmark/results/python/` y `reports/reference/python_reference*` (el script nunca sobrescribe).

## 3. Probar con fotos reales

El motor solo entiende ficheros crudos de 3072 bytes (32×32 píxeles, 3 colores). El script `foto_a_bin.py` convierte tus fotos (JPG, PNG, WebP…): corrige la orientación, recorta al cuadrado central y reduce a 32×32.

### 3.1 Una foto
```bash
python3 scripts/foto_a_bin.py ~/Imágenes/mi_perro.jpg --clasificar
# → mi_perro.jpg: {"prediction": "DOG", "probability": 0.87, "confidence": 0.87}
```
- `prediction`: `DOG` (perro) o `NOT_DOG`.
- `probability`: probabilidad de que sea perro (0 a 1).
- `confidence`: seguridad en la respuesta dada.

### 3.2 Varias fotos a la vez
```bash
python3 scripts/foto_a_bin.py a.jpg b.jpg c.png --batch lote.bin --clasificar
```
Devuelve una lista con un resultado por foto, en el mismo orden.

### 3.3 Fotos de ejemplo con respuesta conocida
```bash
python3 scripts/exportar_muestras.py --n 10     # crea muestras/ con 10 perros y 10 no-perros (la etiqueta va en el nombre)
python3 scripts/foto_a_bin.py muestras/test_0001_perro.png --clasificar
```
Cada imagen tiene una copia `_grande.png` solo para verla (las originales son diminutas, 32×32).

### 3.4 Qué esperar (para no llevarse sorpresas)
- El modelo acierta **~80 %** en las 2.000 fotos de prueba; con pocas fotos el resultado fluctúa mucho (con 20 muestras acertó 14).
- Las fotos de internet o del móvil son más grandes y distintas de las del conjunto de entrenamiento (CIFAR-10: 32×32, un solo objeto centrado). Al reducirlas a 32×32 se pierde mucho detalle: **espera peor acierto** que en el 80 %.
- Un perro pequeño dentro de una escena grande se verá como un borrón tras el recorte. Funciona mejor con el perro ocupando la mayor parte de la foto.
- El modelo solo distingue «perro» / «no perro». No dice qué otra cosa es.
- Sin acceso a internet: todo se ejecuta en local.

### 3.5 Sin el script (a mano)
Si ya tienes un fichero de 3072 bytes: `cd implementations/python && .venv/bin/python -m src.main infer --model runs/reference_001/model_final.tb --image ../../foto.bin` (usa rutas relativas a esa carpeta o absolutas).

## 4. Problemas habituales

| Síntoma | Causa y solución |
|---|---|
| `error: ... una imagen son 3072 bytes` | El fichero no viene de `foto_a_bin.py`. Conviértelo con él. |
| `No module named PIL` | Usa `python3` del sistema (no el del `.venv`), que trae Pillow. |
| El script oficial dice «el equipo no está en reposo» | Cierra programas y espera 2-3 minutos a que baje la carga. `FORCE=1 bash ...` lo salta, pero la carga queda registrada y los números valdrán menos. |
| `ya hay resultados en ...` | Por seguridad no se sobrescribe. Muévelos o bórralos tú. |
