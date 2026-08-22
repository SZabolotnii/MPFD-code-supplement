"""Evaluation: AUC, window scores, bootstrap CIs (window = i.i.d. replicate; block = time series)."""
from __future__ import annotations

import numpy as np

from .basis import Column, features


def auc(s1, s2) -> float:
    v = np.concatenate([np.asarray(s1, float), np.asarray(s2, float)])
    r = v.argsort().argsort() + 1.0
    n1, n2 = len(s1), len(s2)
    a = (r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
    return float(max(a, 1 - a))


def standardizer(*arrays) -> tuple[float, float]:
    allx = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    return float(allx.mean()), float(allx.std())


def window_scores(cols: list[Column], K: np.ndarray, windows: np.ndarray, mu: float, sd: float) -> np.ndarray:
    f = features(cols, (np.asarray(windows, float) - mu) / sd)
    return (f @ np.asarray(K, float)).sum(axis=-1)


def paired_window_bootstrap(s1, s2, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    n = min(len(s1), len(s2))
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = np.array([auc(s1[i], s2[i]) for i in idx])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def moving_block_bootstrap(s1, s2, block: int, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Resample contiguous blocks of windows independently per class (time-ordered scores)."""
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)

    def resample(s):
        nb = int(np.ceil(len(s) / block))
        starts = rng.integers(0, max(len(s) - block, 1), size=nb)
        return np.concatenate([s[st:st + block] for st in starts])[:len(s)]

    bs = np.array([auc(resample(s1), resample(s2)) for _ in range(n_boot)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
