"""Standardized scalar laws as quadrature-able expectations plus samplers.

A `Law` exposes E[f(X)] by deterministic piecewise quadrature (`expect`), a sampler
(`rvs`), the tail index sup{q : E|X|^q < inf} and an optional pdf (None for laws
defined through a transform, e.g. Fleishman). Factory laws are standardized to
mean 0, variance 1 whenever the variance exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate, special, stats
from scipy.optimize import fsolve

Func = Callable[[np.ndarray], np.ndarray]

# The fractional dictionaries have a cusp at the origin, so the grid is refined there;
# a law with a kink of its own declares it through Law.knots (see `laplace`).
_KNOTS = np.array([-np.inf, -50.0, -20.0, -5.0, -1.0, -0.1, -1e-2, -1e-3, -1e-4, -1e-5, 0.0,
                   1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0, 50.0, np.inf])
_ZKNOTS = np.array([-12.0, -8.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])


def _quad_piecewise(g: Func, knots: np.ndarray = _KNOTS) -> float:
    total = 0.0
    for lo, hi in zip(knots[:-1], knots[1:]):
        val, _ = integrate.quad(lambda t: float(g(np.asarray([t]))[0]), lo, hi,
                                limit=800, epsabs=1e-15, epsrel=1e-14)
        total += val
    return float(total)


@dataclass(frozen=True)
class Law:
    name: str
    expect: Callable[[Func], float]
    rvs: Callable[[int, np.random.Generator], np.ndarray]
    tail_index: float
    symmetric: bool
    pdf: Func | None = None
    #: extra quadrature breakpoints (kinks, support edges) on top of the default grid;
    #: an unresolved kink costs several digits, which is visible in C1/C2.
    knots: tuple[float, ...] = ()


def _grid(knots) -> np.ndarray:
    return np.unique(np.concatenate([_KNOTS, np.asarray(knots, float)])) if len(knots) else _KNOTS


def _from_pdf(name: str, pdf: Func, rvs, tail_index: float, symmetric: bool, knots=()) -> Law:
    grid = _grid(knots)
    return Law(name, lambda f: _quad_piecewise(lambda x: f(x) * pdf(x), grid), rvs, tail_index,
               symmetric, pdf, tuple(float(k) for k in knots))


# ----------------------------------------------------------------- basic laws
def gaussian() -> Law:
    return _from_pdf("gaussian", stats.norm.pdf, lambda n, rng: rng.standard_normal(n), np.inf, True)


def laplace() -> Law:
    b = 1.0 / np.sqrt(2.0)
    return _from_pdf("laplace", lambda x: stats.laplace.pdf(x, scale=b),
                     lambda n, rng: rng.laplace(0.0, b, n), np.inf, True, knots=(0.0,))


def student_t(df: float) -> Law:
    s = np.sqrt((df - 2.0) / df) if df > 2 else 1.0
    return _from_pdf(f"t({df:g})", lambda x: stats.t.pdf(np.asarray(x, float) / s, df) / s,
                     lambda n, rng: rng.standard_t(df, n) * s, float(df), True)


# ----------------------------------------------------------------- AEPD (Zhu & Zinde-Walsh)
def _log_kep(p: float) -> float:
    return float(-np.log(2.0) - np.log(p) / p - special.gammaln(1.0 + 1.0 / p))


def _aepd_astar(tau: float, p1: float, p2: float) -> float:
    num = np.log(tau) + _log_kep(p1)
    return float(np.exp(num - np.logaddexp(num, np.log1p(-tau) + _log_kep(p2))))


def _aepd_raw_pdf(z, tau: float, p1: float, p2: float) -> np.ndarray:
    z = np.asarray(z, float)
    astar = _aepd_astar(tau, p1, p2)
    out = np.empty_like(z)
    left = z <= 0
    out[left] = np.exp(np.log(tau) - np.log(astar) + _log_kep(p1) - (np.abs(z[left]) / (2 * astar)) ** p1 / p1)
    out[~left] = np.exp(np.log1p(-tau) - np.log1p(-astar) + _log_kep(p2) - (z[~left] / (2 * (1 - astar))) ** p2 / p2)
    return out


def aepd(tau: float, p_left: float, p_right: float, name: str | None = None) -> Law:
    """Standardized asymmetric exponential-power law; `tau` = left-tail mass (HL-PATP `tau`)."""
    raw = lambda z: _aepd_raw_pdf(z, tau, p_left, p_right)
    astar = _aepd_astar(tau, p_left, p_right)
    m1 = _quad_piecewise(lambda z: z * raw(z))
    m2 = _quad_piecewise(lambda z: z * z * raw(z))
    sd = float(np.sqrt(m2 - m1 * m1))

    def pdf(x):
        return sd * raw(sd * np.asarray(x, float) + m1)

    def rvs(n, rng):
        u = rng.random(n)
        g1 = np.abs(stats.gennorm.rvs(p_left, size=n, random_state=rng))
        g2 = np.abs(stats.gennorm.rvs(p_right, size=n, random_state=rng))
        z = np.where(u < tau, -2 * astar * p_left ** (1 / p_left) * g1,
                     2 * (1 - astar) * p_right ** (1 / p_right) * g2)
        return (z - m1) / sd

    sym = abs(tau - 0.5) < 1e-12 and abs(p_left - p_right) < 1e-12
    # the two exponential-power halves meet at raw z = 0, i.e. at x = -m1/sd after
    # standardisation; for p < 1 the density has an infinite derivative there
    return _from_pdf(name or f"aepd({tau:g},{p_left:g},{p_right:g})", pdf, rvs, np.inf, sym,
                     knots=(-m1 / sd,))


# ----------------------------------------------------------------- Fleishman power method
def fleishman_coefficients(g3: float, g4: float) -> tuple[float, float, float]:
    """(b, c, d) with X = -c + bZ + cZ^2 + dZ^3 having mean 0, var 1, skew g3, excess kurtosis g4."""
    def eqs(v):
        b, c, d = v
        return [b * b + 6 * b * d + 2 * c * c + 15 * d * d - 1.0,
                2 * c * (b * b + 24 * b * d + 105 * d * d + 2) - g3,
                24 * (b * d + c * c * (1 + b * b + 28 * b * d) + d * d * (12 + 48 * b * d + 141 * c * c + 225 * d * d)) - g4]
    sol, _, ier, msg = fsolve(eqs, [1.0, 0.05, 0.03], full_output=True)
    if ier != 1:
        raise ValueError(f"Fleishman solve failed for ({g3}, {g4}): {msg}")
    return tuple(float(v) for v in sol)


def _cubic_real_root(d: float, c: float, b: float, a0) -> np.ndarray:
    """Unique real root of d t^3 + c t^2 + b t + a0 (Cardano); valid where the cubic
    is strictly monotone, i.e. c^2 < 3 b d."""
    a0 = np.asarray(a0, float)
    shift = c / (3.0 * d)
    p = (3.0 * d * b - c * c) / (3.0 * d * d)
    q = (2.0 * c ** 3 - 9.0 * d * c * b + 27.0 * d * d * a0) / (27.0 * d ** 3)
    disc = np.sqrt(np.maximum(q * q / 4.0 + p ** 3 / 27.0, 0.0))
    return np.cbrt(-q / 2.0 + disc) + np.cbrt(-q / 2.0 - disc) - shift


def fleishman(g3: float, g4: float) -> Law:
    b, c, d = fleishman_coefficients(g3, g4)
    g = lambda z: -c + b * z + c * z * z + d * z ** 3
    monotone = d > 0 and c * c < 3.0 * b * d
    pdf = None
    if monotone:
        # x = g(z) is a strictly increasing cubic, so p(x) = phi(z(x)) / g'(z(x))
        def pdf(x):
            z = _cubic_real_root(d, c, b, c - np.asarray(x, float))
            return stats.norm.pdf(z) / (b + 2.0 * c * z + 3.0 * d * z * z)
    return Law(f"fleishman({g3:g},{g4:g})",
               lambda f: _quad_piecewise(lambda z: f(g(z)) * stats.norm.pdf(z), _ZKNOTS),
               lambda n, rng: g(rng.standard_normal(n)), np.inf, g3 == 0.0, pdf)


# ----------------------------------------------------------------- Gaussian mixtures
def gauss_mixture(w, m, s) -> Law:
    w, m, s = (np.asarray(v, float) for v in (w, m, s))
    mean = float(np.sum(w * m))
    var = float(np.sum(w * (s * s + m * m)) - mean * mean)
    sd = np.sqrt(var)
    raw_pdf = lambda x: np.sum(w[:, None] * stats.norm.pdf((np.asarray(x, float)[None, :] - m[:, None]) / s[:, None]) / s[:, None], axis=0)
    pdf = lambda x: sd * raw_pdf(sd * np.asarray(x, float) + mean)

    def rvs(n, rng):
        idx = rng.choice(len(w), size=n, p=w)
        return (rng.normal(m[idx], s[idx]) - mean) / sd

    return _from_pdf(f"mix{tuple(np.round(s, 3))}", pdf, rvs, np.inf, bool(np.all(m == 0.0)))


# ----------------------------------------------------------------- transforms
def affine(law: Law, shift: float = 0.0, scale: float = 1.0, name: str | None = None) -> Law:
    pdf = None if law.pdf is None else (lambda x: law.pdf((np.asarray(x, float) - shift) / scale) / abs(scale))
    return Law(name or f"{law.name}*{scale:g}+{shift:g}",
               lambda f: law.expect(lambda x: f(scale * x + shift)),
               lambda n, rng: scale * law.rvs(n, rng) + shift,
               law.tail_index, law.symmetric and shift == 0.0, pdf,
               tuple(scale * k + shift for k in law.knots))


def reflect(law: Law) -> Law:
    return affine(law, 0.0, -1.0, name=f"reflect({law.name})")


def contaminated(base: Law, eps: float, df: float = 2, scale: float = 6.0) -> Law:
    """(1-eps)*base + eps*(scale * t_df). Tail index becomes min(base, df)."""
    if eps <= 0.0:
        return base
    t = student_t(df)
    pdf = None if base.pdf is None else (lambda x: (1 - eps) * base.pdf(x) + eps * t.pdf(np.asarray(x, float) / scale) / scale)

    def rvs(n, rng):
        x = base.rvs(n, rng)
        mask = rng.random(n) < eps
        k = int(mask.sum())
        if k:
            x = x.copy()
            x[mask] = t.rvs(k, rng) * scale
        return x

    return Law(f"{base.name}+eps{eps:g}t{df:g}",
               lambda f: (1 - eps) * base.expect(f) + eps * t.expect(lambda x: f(scale * x)),
               rvs, min(base.tail_index, float(df)), base.symmetric, pdf, base.knots)


#: public alias -- oracle.py integrates arbitrary targets against the same knots
quad_piecewise = _quad_piecewise
KNOTS = _KNOTS


# ----------------------------------------------------------------- Pearson III (SPL panel)
def pearson3(skew: float) -> Law:
    """Standardized gamma noise of the SPL sweep: mean 0, variance 1, skewness `skew`.

    X = (Z - k)/sqrt(k) with Z ~ Gamma(k, 1), k = 4/skew^2; support x > -sqrt(k).
    `skew = 0` returns the Gaussian limit. Matches verification/p2_intransitivity.noise
    and basis_parity_sweep.logpdf exactly.
    """
    if skew == 0.0:
        return gaussian()
    k = 4.0 / skew ** 2
    edge = -np.sqrt(k)
    pdf = lambda x: np.where(np.asarray(x, float) > edge,
                             stats.gamma.pdf(np.clip((np.asarray(x, float) - edge) * np.sqrt(k), 1e-300, None), k) * np.sqrt(k),
                             0.0)
    knots = np.unique(np.concatenate([[edge], edge + np.array([1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0]),
                                      np.array([-5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 50.0, np.inf])]))
    knots = np.concatenate([[edge], knots[knots > edge]])
    return Law(f"pearson3({skew:g})", lambda f: _quad_piecewise(lambda x: f(x) * pdf(x), knots),
               lambda n, rng: (rng.gamma(k, 1.0, size=n) - k) / np.sqrt(k), np.inf, False, pdf,
               tuple(float(v) for v in knots[np.isfinite(knots)]))
