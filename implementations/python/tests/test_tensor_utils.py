import hashlib

import numpy as np
import pytest

from src.core.tensor_utils import assert_close, compare, read_tb, write_tb


def sample():
    rng = np.random.default_rng(0)
    return {
        "a.f32": rng.standard_normal((2, 3, 4, 5)).astype(np.float32),
        "b.f64": rng.standard_normal((7,)),
        "c.u8": np.arange(5, dtype=np.uint8),            # 5 bytes → relleno a múltiplo de 4
        "d.i32": np.array([-1, 2], dtype=np.int32),
        "e.u32": np.array([7], dtype=np.uint32),
        "f.i16": np.array([-3, 4, 5], dtype=np.int16),
    }


def test_roundtrip_exact(tmp_path):
    t = sample()
    write_tb(tmp_path / "x.tb", t)
    back = read_tb(tmp_path / "x.tb")
    assert sorted(back) == sorted(t)
    for k in t:
        assert back[k].dtype == t[k].dtype and back[k].shape == t[k].shape
        assert np.array_equal(back[k], t[k])


def test_rewrite_is_byte_identical(tmp_path):
    write_tb(tmp_path / "1.tb", sample())
    write_tb(tmp_path / "2.tb", read_tb(tmp_path / "1.tb"))
    h = lambda p: hashlib.sha256((tmp_path / p).read_bytes()).hexdigest()
    assert h("1.tb") == h("2.tb")


def test_layout_matches_spec(tmp_path):
    write_tb(tmp_path / "x.tb", {"w": np.arange(6, dtype=np.float32).reshape(2, 3)})
    b = (tmp_path / "x.tb").read_bytes()
    assert b[:4] == b"AITB" and len(b) == 12 + 72 + 24
    assert int.from_bytes(b[40 + 12 : 48 + 12], "little") == 84          # data_offset
    assert int.from_bytes(b[48 + 12 : 56 + 12], "little") == 6           # n_elements


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        write_tb(tmp_path / "x.tb", {"x" * 32: np.zeros(1, np.float32)})   # nombre de 32 caracteres
    with pytest.raises(ValueError):
        write_tb(tmp_path / "x.tb", {"x": np.zeros((1,) * 5, np.float32)})  # 5 dimensiones
    with pytest.raises(ValueError):
        write_tb(tmp_path / "x.tb", {"x": np.zeros(1, np.float16)})
    (tmp_path / "bad.tb").write_bytes(b"XXXX" + b"\0" * 8)
    with pytest.raises(ValueError):
        read_tb(tmp_path / "bad.tb")
    write_tb(tmp_path / "ok.tb", {"x": np.zeros(4, np.float32)})
    raw = bytearray((tmp_path / "ok.tb").read_bytes())
    raw[12 + 48] = 5                                                          # n_elements incoherente
    (tmp_path / "corrupt.tb").write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_tb(tmp_path / "corrupt.tb")


def test_compare_reports_max_violation_and_nan():
    ok, worst, pos = compare([1.0, 2.0], [1.0, 2.0], 1e-5, 1e-5)
    assert ok and worst <= 0
    ok, worst, pos = compare([1.0, 2.1], [1.0, 2.0], 1e-5, 1e-5)
    assert not ok and pos == (1,) and worst > 0.09
    assert not compare([np.nan], [1.0], 1, 1)[0]
    with pytest.raises(AssertionError):
        assert_close([np.inf], [np.inf], 1, 1)
