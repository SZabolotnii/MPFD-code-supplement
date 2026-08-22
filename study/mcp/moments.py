"""One-sided fractional moment functions nu^+(q), nu^-(q) of a law (population or sample).

Notation follows Ku-HL-PATP: nu_q = nu^+ + nu^-, sigma_q = nu^+ - nu^-.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma

from .laws import Law


class InadmissibleMoment(ValueError):
    def __init__(self, q: float, tail_index: float):
        super().__init__(f"E|X|^{q:g} diverges: tail index {tail_index:g}")
        self.q, self.tail_index = q, tail_index


class OneSidedMoments:
    """Cached nu(q, side) = E[|X|^q 1{side*X > 0}] by quadrature; raises if q >= tail index."""

    def __init__(self, law: Law):
        self.law = law
        self.tail_index = float(law.tail_index)
        self._cache: dict[tuple[float, int], float] = {}

    def nu(self, q: float, side: int) -> float:
        q = float(q)
        if q <= 0:
            raise ValueError("q must be positive")
        if q >= self.tail_index:
            raise InadmissibleMoment(q, self.tail_index)
        key = (round(q, 12), 1 if side > 0 else -1)
        if key not in self._cache:
            if side > 0:
                f = lambda x: np.where(x > 0, np.abs(x) ** q, 0.0)
            else:
                f = lambda x: np.where(x < 0, np.abs(x) ** q, 0.0)
            self._cache[key] = self.law.expect(f)
        return self._cache[key]

    def nu_total(self, q: float) -> float:
        return self.nu(q, +1) + self.nu(q, -1)

    def sigma(self, q: float) -> float:
        return self.nu(q, +1) - self.nu(q, -1)


class SampleOneSidedMoments:
    """Plug-in nu^+/-(q) from a sample; always finite (tail_index = inf)."""

    def __init__(self, x: np.ndarray):
        self.x = np.asarray(x, float).ravel()
        self.n = self.x.size
        self.tail_index = np.inf
        self._ax = np.abs(self.x)
        self._pos = self.x > 0
        self._neg = self.x < 0

    def nu(self, q: float, side: int) -> float:
        mask = self._pos if side > 0 else self._neg
        return float(np.sum(self._ax[mask] ** q) / self.n)

    def nu_total(self, q: float) -> float:
        return self.nu(q, +1) + self.nu(q, -1)

    def sigma(self, q: float) -> float:
        return self.nu(q, +1) - self.nu(q, -1)


def hill_tail_index(x: np.ndarray, frac: float = 0.05) -> float:
    """Hill estimator on |x| with k = frac * n upper order statistics (k >= 10)."""
    a = np.sort(np.abs(np.asarray(x, float).ravel()))[::-1]
    k = max(int(frac * a.size), 10)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


# ----------------------------------------------------------------- closed forms (test oracles)
def gaussian_nu_plus(q: float) -> float:
    return float(2 ** (q / 2 - 1) * gamma((q + 1) / 2) / np.sqrt(np.pi))


def laplace_nu_plus(q: float) -> float:
    b = 1.0 / np.sqrt(2.0)
    return float(0.5 * b ** q * gamma(q + 1))


def student_t_nu_plus(q: float, df: float) -> float:
    s = np.sqrt((df - 2.0) / df)
    return float(0.5 * s ** q * df ** (q / 2) * gamma((q + 1) / 2) * gamma((df - q) / 2)
                 / (np.sqrt(np.pi) * gamma(df / 2)))
