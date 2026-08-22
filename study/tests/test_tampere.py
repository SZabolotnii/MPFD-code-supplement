import sys
from pathlib import Path

import numpy as np
import pytest

from mcp.data import tampere as T


def _write_fake(path: Path, n: int, rng):
    iq = (rng.normal(0, 2000, (n, 2))).astype("<i2")
    iq[2 * n // 3:] *= 4                  # last third is a "burst" (median window power stays low)
    path.write_bytes(iq.tobytes())


def test_read_iq_and_gate(tmp_path):
    p = tmp_path / "Fake_drone_2G.bin"
    _write_fake(p, 40_000, np.random.default_rng(0))
    assert T.complex_samples(p) == 40_000
    z = T.read_iq(p, 10, 5)
    assert z.dtype == np.complex64 and z.shape == (5,) and np.all(np.abs(z) < 1.0)
    w = T.gated_real_windows(p, N=256, gate=3.0, start_frac=0.0, stop_frac=1.0, max_windows=1000, chunk_windows=32)
    assert w.ndim == 2 and w.shape[1] == 256
    assert 45 <= w.shape[0] <= 60          # ~ last third of 156 windows passes the gate


def test_bridge_sdr_dron_reader(tmp_path):
    src = Path("/Users/docua/Project/Research/SDR_Dron/src")
    if not src.exists():
        pytest.skip("SDR_Dron not available")
    sys.path.insert(0, str(src))
    iq_reader = pytest.importorskip("sdr_dron.io.iq_reader")
    p = tmp_path / "Fake_drone_2G.bin"
    _write_fake(p, 10_000, np.random.default_rng(1))
    ours = T.read_iq(p, 100, 500)
    theirs = iq_reader.read_iq_window(p, start_sample=100, sample_count=500, normalize="int16")
    assert np.allclose(ours, theirs)
