"""Dictionaries of fractional-power columns.

Every column is c_plus*|x|^a on x>0 and c_minus*|x|^a on x<0, so population moments
reduce to the one-sided fractional moments nu^+/-(q) at q = a_i and a_i + a_j.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_COEF = {"linear": (1.0, -1.0), "abs": (1.0, 1.0), "signed": (1.0, -1.0),
         "half_plus": (1.0, 0.0), "half_minus": (0.0, 1.0)}
_PARITY = {"linear": "odd", "abs": "even", "signed": "odd", "half_plus": "none", "half_minus": "none"}


@dataclass(frozen=True)
class Column:
    kind: str
    a: float

    def __post_init__(self):
        if self.kind not in _COEF:
            raise ValueError(f"unknown column kind {self.kind!r}")
        if self.kind == "linear" and self.a != 1.0:
            raise ValueError("linear column has exponent 1")
        if self.a <= 0:
            raise ValueError("exponent must be positive")

    @property
    def coef(self) -> tuple[float, float]:
        return _COEF[self.kind]

    @property
    def parity(self) -> str:
        return _PARITY[self.kind]

    @property
    def label(self) -> str:
        return {"linear": "x", "abs": f"|x|^{self.a:g}", "signed": f"sgn(x)|x|^{self.a:g}",
                "half_plus": f"(x+)^{self.a:g}", "half_minus": f"(x-)^{self.a:g}"}[self.kind]

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        ax = np.abs(x)
        if self.kind == "linear":
            return x
        if self.kind == "abs":
            return ax ** self.a
        if self.kind == "signed":
            return np.sign(x) * ax ** self.a
        if self.kind == "half_plus":
            return np.where(x > 0, ax ** self.a, 0.0)
        return np.where(x < 0, ax ** self.a, 0.0)


LINEAR = Column("linear", 1.0)


def p_i(i: int, alpha: float) -> float:
    """Frozen PATP exponent map (arXiv:2605.14610)."""
    return 1.0 / i + (4.0 - i - 3.0 / i) * alpha + (2.0 * i - 4.0 + 2.0 / i) * alpha ** 2


def patp(alpha: float, s: int) -> list[Column]:
    return [LINEAR] + [Column("signed", p_i(i, alpha)) for i in range(2, s + 1)]


def mixed(A) -> list[Column]:
    cols = [LINEAR]
    for a in A:
        cols += [Column("abs", float(a)), Column("signed", float(a))]
    return cols


def halfline(A_plus, A_minus) -> list[Column]:
    return [LINEAR] + [Column("half_plus", float(a)) for a in A_plus] + [Column("half_minus", float(b)) for b in A_minus]


def classical(s: int) -> list[Column]:
    return [LINEAR] + [Column("abs" if k % 2 == 0 else "signed", float(k)) for k in range(2, s + 1)]


def free_signed(A) -> list[Column]:
    return [LINEAR] + [Column("signed", float(a)) for a in A]


def features(cols, x) -> np.ndarray:
    x = np.asarray(x, float)
    return np.stack([c(x) for c in cols], axis=-1)


def max_exponent(cols) -> float:
    return max(c.a for c in cols)


def exponent_ceiling(alpha_min: float) -> float:
    return alpha_min / 2.0


def admissible(cols, alpha_min: float) -> bool:
    return 2.0 * max_exponent(cols) < alpha_min
