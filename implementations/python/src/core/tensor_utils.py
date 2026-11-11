"""Formato TensorBundle v1 (docs/architecture.md §5) y comparación con tolerancias."""
import struct
from pathlib import Path

import numpy as np

MAGIC = b"AITB"
VERSION = 1
HEADER = struct.Struct("<4sHHI")            # magic, version, flags, n_tensors  (12 B)
ENTRY = struct.Struct("<32sHBBIQQ4I")       # 72 B por entrada de índice
assert HEADER.size == 12 and ENTRY.size == 72

DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<f8"), 2: np.dtype("u1"),
          3: np.dtype("<i4"), 4: np.dtype("<u4"), 5: np.dtype("<i2")}
CODES = {dt: code for code, dt in DTYPES.items()}


def _pad4(n):
    return (n + 3) // 4 * 4


def write_tb(path, tensors):
    """Escribe un TensorBundle. Índice en orden alfabético de nombre; datos row-major LE."""
    names = sorted(tensors)
    index_end = HEADER.size + ENTRY.size * len(names)
    entries, blocks, offset = [], [], index_end
    for name in names:
        arr = np.ascontiguousarray(tensors[name])
        dt = arr.dtype.newbyteorder("<")
        if dt not in CODES:
            raise ValueError(f"{name}: dtype no soportado {arr.dtype}")
        if not 1 <= arr.ndim <= 4:
            raise ValueError(f"{name}: ndim {arr.ndim} fuera de [1, 4]")
        raw_name = name.encode("ascii")
        if len(raw_name) > 31:
            raise ValueError(f"{name}: nombre de más de 31 caracteres")
        shape = list(arr.shape) + [0] * (4 - arr.ndim)
        data = arr.astype(dt, copy=False).tobytes()
        entries.append(ENTRY.pack(raw_name, CODES[dt], arr.ndim, 0, 0, offset, arr.size, *shape))
        blocks.append(data + b"\x00" * (_pad4(len(data)) - len(data)))
        offset += _pad4(len(data))
    Path(path).write_bytes(HEADER.pack(MAGIC, VERSION, 0, len(names)) + b"".join(entries) + b"".join(blocks))


def read_tb(path):
    """Lee un TensorBundle → dict nombre → ndarray. Valida coherencia; falla ante cualquier anomalía."""
    buf = Path(path).read_bytes()
    if len(buf) < HEADER.size:
        raise ValueError("fichero demasiado corto")
    magic, version, _flags, n = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or version != 1:
        raise ValueError(f"cabecera no válida: {magic!r} v{version}")
    out = {}
    for i in range(n):
        raw, code, ndim, _r1, _r2, off, n_el, *shape = ENTRY.unpack_from(buf, HEADER.size + i * ENTRY.size)
        name = raw.split(b"\x00", 1)[0].decode("ascii")
        if code not in DTYPES or not 1 <= ndim <= 4:
            raise ValueError(f"{name}: dtype {code} / ndim {ndim} no válidos")
        shape = tuple(shape[:ndim])
        if int(np.prod(shape, dtype=np.uint64)) != n_el:
            raise ValueError(f"{name}: n_elements {n_el} no coincide con shape {shape}")
        dt = DTYPES[code]
        if off % 4 or off + n_el * dt.itemsize > len(buf):
            raise ValueError(f"{name}: data_offset {off} inválido")
        out[name] = np.frombuffer(buf, dtype=dt, count=n_el, offset=off).reshape(shape).copy()
    return out


def check_shape(name, arr, expected):
    if tuple(arr.shape) != tuple(expected):
        raise ValueError(f"{name}: forma {tuple(arr.shape)}, se esperaba {tuple(expected)}")


def compare(actual, expected, atol, rtol):
    """Compara con |a−b| ≤ atol + rtol·|b|. Devuelve (ok, violación máxima, posición).

    La violación es max(|a−b| − (atol + rtol·|b|)); NaN/Inf en cualquiera de los dos es fallo inmediato.
    """
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return False, float("inf"), ("forma", a.shape, b.shape)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False, float("inf"), "NaN/Inf"
    excess = np.abs(a - b) - (atol + rtol * np.abs(b))
    pos = np.unravel_index(np.argmax(excess), excess.shape) if excess.size else ()
    worst = float(excess[pos]) if excess.size else 0.0
    return worst <= 0.0, worst, pos


def assert_close(actual, expected, atol, rtol, what=""):
    ok, worst, pos = compare(actual, expected, atol, rtol)
    if not ok:
        raise AssertionError(f"{what}: fuera de tolerancia (atol={atol}, rtol={rtol}); exceso máximo {worst:.3e} en {pos}")
