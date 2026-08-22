"""Deflection-optimal (Kunchenko, lambda = 1/2) projection of the LLR on a dictionary.

F = C_m + C_n (feature covariances), Y = mu_n - mu_m, F K = Y, J = Y' F^{-1} Y.
All population quantities come from one-sided fractional moments only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .basis import Column, features
from .laws import Law

COND_LIMIT = 1e6


@dataclass
class ClassMoments:
    mu: np.ndarray
    C: np.ndarray


def population_moments(cols: list[Column], osm) -> ClassMoments:
    k = len(cols)
    mu = np.empty(k)
    R = np.empty((k, k))
    for i, ci in enumerate(cols):
        cp, cm = ci.coef
        mu[i] = cp * osm.nu(ci.a, +1) + cm * osm.nu(ci.a, -1)
        for j, cj in enumerate(cols):
            dp, dm = cj.coef
            q = ci.a + cj.a
            R[i, j] = cp * dp * osm.nu(q, +1) + cm * dm * osm.nu(q, -1)
    return ClassMoments(mu, R - np.outer(mu, mu))


def sample_moments(cols: list[Column], x: np.ndarray) -> ClassMoments:
    f = features(cols, np.asarray(x, float).ravel())
    mu = f.mean(axis=0)
    C = np.atleast_2d(np.cov(f, rowvar=False, bias=True))
    return ClassMoments(mu, C)


@dataclass
class Deflection:
    K: np.ndarray
    J: float
    F: np.ndarray
    Y: np.ndarray
    cond: float


def deflection(cm: ClassMoments, cn: ClassMoments) -> Deflection:
    """Solve F K = Y for the deflection-optimal projection; no regularisation by design --
    a singular/ill-conditioned F is reported via `cond` (and NaN K/J), never patched with a ridge."""
    F = cm.C + cn.C
    Y = cn.mu - cm.mu
    cond = float(np.linalg.cond(F))
    try:
        K = np.linalg.solve(F, Y)
    except np.linalg.LinAlgError:
        K = np.full_like(Y, np.nan)
    J = float(Y @ K) if np.all(np.isfinite(K)) else float("nan")
    return Deflection(K, J, F, Y, cond if np.isfinite(cond) else float("inf"))


def schur_gain(cm: ClassMoments, cn: ClassMoments) -> tuple[float, float, float]:
    """Split the last column off: J = J_head + Ytilde^2 / S (L4)."""
    F = cm.C + cn.C
    Y = cn.mu - cm.mu
    A, b, d = F[:-1, :-1], F[:-1, -1], F[-1, -1]
    y, eta = Y[:-1], Y[-1]
    Ay = np.linalg.solve(A, y)
    Ab = np.linalg.solve(A, b)
    return float(y @ Ay), float(eta - b @ Ay), float(d - b @ Ab)


def _sub(cm: ClassMoments, idx: list[int]) -> ClassMoments:
    return ClassMoments(cm.mu[idx], cm.C[np.ix_(idx, idx)])


def parity_split(cols: list[Column], cm: ClassMoments, cn: ClassMoments) -> dict:
    """J for the full dictionary and for its even-only / odd-only sub-dictionaries; columns
    with parity == "none" (half-line columns) belong to neither split."""
    even = [i for i, c in enumerate(cols) if c.parity == "even"]
    odd = [i for i, c in enumerate(cols) if c.parity == "odd"]
    out = {"J_full": deflection(cm, cn).J}
    out["J_even"] = deflection(_sub(cm, even), _sub(cn, even)).J if even else 0.0
    out["J_odd"] = deflection(_sub(cm, odd), _sub(cn, odd)).J if odd else 0.0
    return out


def j_halfline_s3(osm_m, osm_n, a: float, b: float) -> float:
    """Explicit form of J for {x, (x+)^a, (x-)^b} from eight one-sided moments per class."""
    def stats(o):
        s1, nu2 = o.sigma(1.0), o.nu_total(2.0)
        mp, mm = o.nu(a, +1), o.nu(b, -1)
        rpp, rmm = o.nu(2 * a, +1), o.nu(2 * b, -1)
        rxp, rxm = o.nu(a + 1, +1), -o.nu(b + 1, -1)
        mu = np.array([s1, mp, mm])
        R = np.array([[nu2, rxp, rxm], [rxp, rpp, 0.0], [rxm, 0.0, rmm]])
        return mu, R - np.outer(mu, mu)

    mu_m, C_m = stats(osm_m)
    mu_n, C_n = stats(osm_n)
    F = C_m + C_n
    Y = mu_n - mu_m
    # adjugate of the symmetric 3x3 metric -- the manuscript's closed form
    c00 = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
    c01 = -(F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
    c02 = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
    c11 = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
    c12 = -(F[0, 0] * F[2, 1] - F[0, 1] * F[2, 0])
    c22 = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    det = F[0, 0] * c00 + F[0, 1] * c01 + F[0, 2] * c02
    adj = np.array([[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]])
    return float(Y @ adj @ Y / det)


def triangular_discrimination(law_m: Law, law_n: Law) -> float:
    if law_m.pdf is None or law_n.pdf is None:
        raise ValueError("triangular discrimination needs densities")
    knots = np.array([-np.inf, -50.0, -20.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 20.0, 50.0, np.inf])

    def g(t):
        pm, pn = float(law_m.pdf(np.asarray([t]))[0]), float(law_n.pdf(np.asarray([t]))[0])
        return 0.0 if pm + pn <= 0 else (pn - pm) ** 2 / (pm + pn)

    return float(sum(integrate.quad(g, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
                     for lo, hi in zip(knots[:-1], knots[1:])))


def j_infinity(law_m: Law, law_n: Law) -> tuple[float, float]:
    d = triangular_discrimination(law_m, law_n)
    return d, (np.inf if d >= 2 else 2 * d / (2 - d))


# ================================================================= lambda-family (TSP, Theorem 1)
# F_lambda = lambda C_m + (1-lambda) C_n, K = F_lambda^{-1} Y, J = Y' F_lambda^{-1} Y.
# The Letter's `deflection` above uses F = C_m + C_n = 2 F_{1/2}, so J_letter = J_{1/2}/2.

@dataclass
class LambdaSolution:
    lam: float
    K: np.ndarray          # direction from the moment normal system F_lam K = Y
    J: float               # Y' F_lam^{-1} Y
    k: np.ndarray          # the L2(w_lam) projection coefficient: K / (1 + lam(1-lam) J)
    scale: float
    F: np.ndarray
    Y: np.ndarray
    cond: float


def f_lambda(cm: ClassMoments, cn: ClassMoments, lam: float) -> np.ndarray:
    return lam * cm.C + (1.0 - lam) * cn.C


def sherman_morrison_scale(J: float, lam: float) -> float:
    """Projection scale of Theorem 1: (Cov_w phi)^{-1} Y = K / (1 + lam(1-lam) J)."""
    return 1.0 / (1.0 + lam * (1.0 - lam) * J)


def solve_lambda(cm: ClassMoments, cn: ClassMoments, lam: float) -> LambdaSolution:
    F = f_lambda(cm, cn, lam)
    Y = cn.mu - cm.mu
    cond = float(np.linalg.cond(F))
    try:
        K = np.linalg.solve(F, Y)
    except np.linalg.LinAlgError:
        K = np.full_like(Y, np.nan)
    J = float(Y @ K) if np.all(np.isfinite(K)) else float("nan")
    scale = sherman_morrison_scale(J, lam)
    return LambdaSolution(float(lam), K, J, scale * K, scale, F, Y,
                          cond if np.isfinite(cond) else float("inf"))


def cov_mixture(mus: np.ndarray, Cs: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Metric of the M-ary rule: Cov_w phi = S_W + S_B (P4), plus the mixture mean."""
    priors = np.asarray(priors, float)
    mu_bar = priors @ mus
    d = mus - mu_bar
    S_W = np.einsum("m,mij->ij", priors, Cs)
    S_B = np.einsum("m,mi,mj->ij", priors, d, d)
    return S_W, S_B, mu_bar


@dataclass
class MarySolution:
    k: np.ndarray          # (M, s) coefficients of the centered columns
    a0: np.ndarray         # (M,) constant terms; ghat_m = a0_m + k_m . phi
    J_M: float             # sum_m ||Pi_s r_m||^2_w
    cov: np.ndarray        # the shared metric S_W + S_B
    cond: float


def mary_projection(mus: np.ndarray, Cs: np.ndarray, priors) -> MarySolution:
    """Least-squares projection of the posteriors r_m on span{1, phi} in L2(w).

    Bordered system G g_m = b_m with psi = (1, phi) reduces (Supplementary, P2) to
    (Cov_w phi) k_m = pi_m (mu^m - mu_bar),  a0_m = pi_m - mu_bar . k_m.
    """
    priors = np.asarray(priors, float)
    mus = np.asarray(mus, float)
    S_W, S_B, mu_bar = cov_mixture(mus, np.asarray(Cs, float), priors)
    cov = S_W + S_B
    rhs = (priors[:, None] * (mus - mu_bar)).T                 # (s, M)
    cond = float(np.linalg.cond(cov))
    k = np.linalg.solve(cov, rhs).T                            # (M, s)
    a0 = priors - k @ mu_bar
    J_M = float(np.sum(priors ** 2) + np.einsum("mi,mi->", rhs.T, k))
    return MarySolution(k, a0, J_M, cov, cond if np.isfinite(cond) else float("inf"))


def mary_scores(sol: MarySolution, Phi: np.ndarray) -> np.ndarray:
    """ghat_m(x) for every row of the feature matrix Phi; rows sum to 1 (P3)."""
    return np.asarray(Phi, float) @ sol.k.T + sol.a0


def whiten_moments(mus: np.ndarray, Cs: np.ndarray, priors) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reparametrise the dictionary so that the mixture metric is the identity.

    phi -> T (phi - mu_bar) with T = D^{-1/2} V' from the eigendecomposition of
    Cov_w phi. Every quantity of section 3 (the rule, J_M, the risk) is invariant
    under an invertible affine change of dictionary; only the conditioning changes.
    Returns (mus_w, Cs_w, T, mu_bar).
    """
    priors = np.asarray(priors, float)
    mus = np.asarray(mus, float)
    Cs = np.asarray(Cs, float)
    S_W, S_B, mu_bar = cov_mixture(mus, Cs, priors)
    w, V = np.linalg.eigh(S_W + S_B)
    T = (V @ np.diag(np.maximum(w, 1e-300) ** -0.5) @ V.T)
    return (mus - mu_bar) @ T.T, np.einsum("ij,mjk,lk->mil", T, Cs, T), T, mu_bar
