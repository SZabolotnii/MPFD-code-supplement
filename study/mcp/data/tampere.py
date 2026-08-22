"""Tampere/Zenodo 10.5281/zenodo.4264467 drone-RF recordings: int16 interleaved IQ.

One recording per emitter; the only genuine remote-vs-drone pair is
Parrot_mambo_control_2G vs Parrot_mambo_video_2G. Resampling unit for CIs is a
moving block of windows (time-ordered); n_recordings = 1 per class.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

RAW = Path(os.environ.get("MPFD_TAMPERE_RAW", "/Users/docua/Project/Research/SDR_Dron/data/raw"))
CONTROL = RAW / "Parrot_mambo_control_2G.bin"
VIDEO = RAW / "Parrot_mambo_video_2G.bin"


def available() -> bool:
    return CONTROL.exists() and VIDEO.exists()


def complex_samples(path: str | Path) -> int:
    return Path(path).stat().st_size // 4


def read_iq(path: str | Path, start: int, count: int) -> np.ndarray:
    raw = np.memmap(Path(path), dtype="<i2", mode="r", offset=start * 4, shape=(count * 2,))
    pairs = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    return ((pairs[:, 0] + 1j * pairs[:, 1]) / np.float32(32768.0)).astype(np.complex64)


def gated_real_windows(path: str | Path, N: int = 1024, gate: float = 3.0, start_frac: float = 0.0,
                       stop_frac: float = 0.5, max_windows: int = 20000, chunk_windows: int = 4096) -> np.ndarray:
    """Real part of IQ windows whose mean power > gate * median window power of the scanned range."""
    total = complex_samples(path)
    lo, hi = int(total * start_frac), int(total * stop_frac)
    powers, starts = [], []
    for s in range(lo, hi - N + 1, N * chunk_windows):
        count = min(N * chunk_windows, hi - s) // N * N
        if count <= 0:
            break
        z = read_iq(path, s, count).reshape(-1, N)
        powers.append(np.mean(np.abs(z) ** 2, axis=1))
        starts.append(s + N * np.arange(z.shape[0]))
    powers, starts = np.concatenate(powers), np.concatenate(starts)
    keep = starts[powers > gate * np.median(powers)]
    if keep.size > max_windows:
        keep = keep[np.linspace(0, keep.size - 1, max_windows).astype(int)]   # evenly thinned, time order kept
    return np.stack([read_iq(path, int(s), N).real for s in keep]) if keep.size else np.empty((0, N))
