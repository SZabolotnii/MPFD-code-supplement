"""Numerical engine for the stage-6 study of hypothesis H1 v4 (see DESIGN.md).

Model: clean Gaussian classes, signed-Pareto contaminant with P(|C| > x) = x^-3 (x >= 1),
a three-column dictionary with one inadmissible top column. Three jobs:

- `fit`            the detector of paper/reframe/main.tex (per-class covariance, divisor n);
- `population_*`   K^sub and test means by quadrature + closed-form Pareto moments;
- `h_sample`       the Poisson-series law of H = N/D (Section 1 of the note);
- `CFEngine`       the exact characteristic function of one window pair and the
                   Gil-Pelaez AUC_N for any N.
"""
from __future__ import annotations

import json
import math
import zlib

import numpy as np
from scipy import integrate, interpolate, optimize, special

ALPHA = 3.0
G_PLUS = (0.3, 1.3)
G_MINUS = (0.0, 1.0)

DICTS = {
    "oo": [("e", 0.5), ("o", 1.0), ("o", None)],
    "oe": [("e", 0.5), ("o", 1.0), ("e", None)],
    "ee": [("o", 0.5), ("e", 1.2), ("e", None)],
    "eo": [("o", 0.5), ("e", 1.2), ("o", None)],
}


def columns(name: str, astar: float) -> list[tuple[str, float]]:
    return [(p, astar if a is None else a) for p, a in DICTS[name]]


def same_parity(cols) -> bool:
    return cols[-1][0] == cols[-2][0]


def seed_of(params: dict, salt: str = "") -> int:
    return zlib.crc32((json.dumps(params, sort_keys=True) + salt).encode())


# --------------------------------------------------------------------------- data

def phi(cols, x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    sg = np.sign(x)
    out = np.empty((x.size, len(cols)))
    for j, (p, a) in enumerate(cols):
        v = ax ** a
        out[:, j] = v if p == "e" else sg * v
    return out


def draw(rng, n: int, eps: float, clean: tuple[float, float], pplus: float) -> np.ndarray:
    m = rng.binomial(n, eps) if eps > 0 else 0
    x = np.empty(n)
    x[: n - m] = rng.normal(clean[0], clean[1], n - m)
    if m:
        mag = (1.0 - rng.random(m)) ** (-1.0 / ALPHA)
        sgn = np.where(rng.random(m) < pplus, 1.0, -1.0)
        x[n - m:] = sgn * mag
    return x


def fit(cols, xp: np.ndarray, xm: np.ndarray) -> tuple[np.ndarray, float]:
    """K-hat = F^-1 Y with F = Sigma_+ + Sigma_- (divisor n); returns (K, cond of equilibrated F)."""
    Pp, Pm = phi(cols, xp), phi(cols, xm)
    mp, mm = Pp.mean(0), Pm.mean(0)
    Sp = (Pp - mp).T @ (Pp - mp) / len(xp)
    Sm = (Pm - mm).T @ (Pm - mm) / len(xm)
    F = Sp + Sm
    Y = mp - mm
    d = np.sqrt(np.diag(F))
    Fe = F / np.outer(d, d)
    K = np.linalg.solve(Fe, Y / d) / d
    return K, float(np.linalg.cond(Fe))


# --------------------------------------------------------------------------- population moments

def _gauss_moment(parity: str, q: float, mu: float, sd: float) -> float:
    pdf = lambda x: math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    pos = integrate.quad(lambda x: x ** q * pdf(x), 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    neg = integrate.quad(lambda x: x ** q * pdf(-x), 0, np.inf, epsabs=1e-13, epsrel=1e-12)[0]
    return pos + neg if parity == "e" else pos - neg


def _pareto_moment(parity: str, q: float, pplus: float) -> float:
    if q >= ALPHA:
        raise ValueError(f"moment of order {q} is infinite")
    m = ALPHA / (ALPHA - q)
    return m if parity == "e" else (2 * pplus - 1) * m


def _mix_moment(parity, q, eps, clean, pplus):
    return (1 - eps) * _gauss_moment(parity, q, *clean) + eps * _pareto_moment(parity, q, pplus)


def _prod(c1, c2):
    return ("e" if c1[0] == c2[0] else "o", c1[1] + c2[1])


def population_mean(cols, eps, clean, pplus) -> np.ndarray:
    return np.array([_mix_moment(p, a, eps, clean, pplus) for p, a in cols])


def population_cov(cols, eps, clean, pplus) -> np.ndarray:
    m = population_mean(cols, eps, clean, pplus)
    s = len(cols)
    C = np.empty((s, s))
    for i in range(s):
        for j in range(s):
            C[i, j] = _mix_moment(*_prod(cols[i], cols[j]), eps, clean, pplus) - m[i] * m[j]
    return C


def ksub(cols, eps_tr: tuple[float, float], pplus: float) -> np.ndarray:
    L = cols[:-1]
    F = population_cov(L, eps_tr[0], G_PLUS, pplus) + population_cov(L, eps_tr[1], G_MINUS, pplus)
    Y = population_mean(L, eps_tr[0], G_PLUS, pplus) - population_mean(L, eps_tr[1], G_MINUS, pplus)
    return np.linalg.solve(F, Y)


def test_mean_diff_L(cols, eps_te: tuple[float, float], pplus: float) -> np.ndarray:
    L = cols[:-1]
    return population_mean(L, eps_te[0], G_PLUS, pplus) - population_mean(L, eps_te[1], G_MINUS, pplus)


# --------------------------------------------------------------------------- H = N / D

def h_sample(cols, pplus: float, size: int, rng, terms: int = 5000, chunk: int = 1000,
             campbell: bool = False) -> np.ndarray:
    """Draws of H = sum phi_l* phi_t(J_k) / sum phi_t(J_k)^2, J_k = xi_k Gamma_k^(-1/alpha).

    For opposite parity the tail beyond `terms` is its mean plus a normal term. Run 1 used the
    variance given the points, 4 p+ p- times the tail integral (`campbell=False`, kept so that the
    stored run-1 results reproduce). The full variance of a Poisson tail sum is the integral of
    E[(xi t^-p1)^2] = t^-2p1, factor 1 (`campbell=True`). They agree at p+ = 1/2.
    `diag_h_tail.py` measures the Kolmogorov distance between the two.
    """
    aL, at = cols[-2][1], cols[-1][1]
    p1, p2 = (at + aL) / ALPHA, 2 * at / ALPHA
    same = same_parity(cols)
    out = np.empty(size)
    for s in range(0, size, chunk):
        b = min(chunk, size - s)
        G = np.cumsum(rng.exponential(1.0, (b, terms)), axis=1)
        D = (G ** -p2).sum(1) + G[:, -1] ** (1 - p2) / (p2 - 1)
        if same:
            Nn = (G ** -p1).sum(1) + G[:, -1] ** (1 - p1) / (p1 - 1)
        else:
            xi = np.where(rng.random((b, terms)) < pplus, 1.0, -1.0)
            Nn = (xi * G ** -p1).sum(1)
            GK = G[:, -1]
            Nn += (2 * pplus - 1) * GK ** (1 - p1) / (p1 - 1)
            fac = 1.0 if campbell else 1 - (2 * pplus - 1) ** 2
            var = fac * GK ** (1 - 2 * p1) / (2 * p1 - 1)
            Nn += np.sqrt(var) * rng.standard_normal(b)
        out[s:s + b] = Nn / D
    return out


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(Fa - Fb).max())


# --------------------------------------------------------------------------- stable law

def rho(gam: float, beta: float) -> float:
    return 0.5 + math.atan(beta * math.tan(math.pi * gam / 2)) / (math.pi * gam)


def stable_scale_pow(gam: float) -> float:
    """sigma^gamma of Z_{gamma,beta} whose Levy tails total y^-gamma."""
    return special.gamma(1 - gam) * math.cos(math.pi * gam / 2)


def stable_cdf_sym(x: float, gam: float) -> float:
    """P(Z <= x) for the symmetric Z_{gamma,0} by Gil-Pelaez (independent of the window engine)."""
    if x == 0:
        return 0.5
    c = stable_scale_pow(gam)
    ax = abs(x)
    e = lambda u: math.exp(-c * (u / ax) ** gam)  # u = t |x|
    head = integrate.quad(lambda u: e(u) * math.sin(u) / u, 0, 1, epsabs=1e-13)[0]
    tail = integrate.quad(lambda u: e(u) / u, 1, np.inf, weight="sin", wvar=1.0, epsabs=1e-13)[0]
    return 0.5 + math.copysign(head + tail, x) / math.pi


def lambda_mid(gam: float, level: float) -> float:
    """lambda with P(1 + lambda Z_{gam,0} > 0) = level, i.e. P(Z <= -1/lambda) = 1 - level."""
    q = optimize.brentq(lambda x: stable_cdf_sym(x, gam) - (1 - level), -1e4, -1e-6, xtol=1e-12)
    return -1.0 / q


# --------------------------------------------------------------------------- CF engine

_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _one_minus_eix(theta: np.ndarray) -> np.ndarray:
    return 2 * np.sin(theta / 2) ** 2 - 1j * np.sin(theta)


def _log1m(a: np.ndarray) -> np.ndarray:
    """log(1 - a), accurate for small |a|."""
    out = np.empty_like(a)
    small = np.abs(a) < 1e-3
    z = -a[small]
    out[small] = z - z ** 2 / 2 + z ** 3 / 3 - z ** 4 / 4 + z ** 5 / 5 - z ** 6 / 6
    out[~small] = np.log(1 - a[~small])
    return out


class _HighEndShort(RuntimeError):
    """|psi_D(t)|^N has not decayed by the end of the t-grid."""


class CFEngine:
    """Exact CF of the score g(x) = k . phi(x) under G_+, G_- and the two Pareto half-laws."""

    def __init__(self, cols, k: np.ndarray, n_gauss: int = 3000,
                 s_min: float = -45 * math.log(10), per_decade: int = 60, pareto: bool = True):
        self.cols, self.k = cols, np.asarray(k, float)
        self.g_sd = math.sqrt(0.5 * (self._clean_var(G_PLUS) + self._clean_var(G_MINUS)))
        self.pareto, self.per_decade = pareto, per_decade
        t_max = 60.0 / self.g_sd
        nd = int((math.log10(t_max) - s_min / math.log(10)) * per_decade) + 1
        w, wt = np.polynomial.legendre.leggauss(n_gauss)
        self._w, self._wt = 0.5 * (w + 1), 0.5 * wt
        self.s = np.linspace(s_min, math.log(t_max), nd)
        self.t = np.exp(self.s)
        self.UGp, self.UGm, self.UQp, self.UQm = self._values(self.t)

    def _values(self, t: np.ndarray):
        UGp = self._u_gauss(G_PLUS, t)
        UGm = self._u_gauss(G_MINUS, t)
        if self.pareto:
            UQp = np.array([self._u_pareto(x, +1.0) for x in t])
            UQm = np.array([self._u_pareto(x, -1.0) for x in t])
        else:  # clean windows only (eps_te must then be (0, 0))
            UQp = UQm = np.full(t.size, np.nan + 0j)
        return UGp, UGm, UQp, UQm

    def auc(self, eps_te: tuple[float, float], pplus: float, Ns) -> np.ndarray:
        """AUC_N for each N; nan where |psi_D|^N has not decayed by the end of the t-grid.

        Extending the grid is not attempted: at large t the Pareto panel quadrature needs
        millions of panels per point. The nan entries are small N only (|psi_D|^N decays
        faster for larger N) and are counted by the caller.
        """
        L = self.logcf(eps_te, pplus)
        out = np.empty(len(Ns))
        for i, N in enumerate(Ns):
            try:
                out[i] = self.auc_curve(L, [N])[0]
            except _HighEndShort:
                out[i] = np.nan
        return out

    # score and its u-derivative on x = side * e^u
    def g(self, x):
        return phi(self.cols, np.atleast_1d(x)) @ self.k

    def _clean_var(self, clean):
        mu, sd = clean
        x = mu + sd * np.linspace(-12, 12, 200001)
        w = np.exp(-0.5 * ((x - mu) / sd) ** 2)
        w /= w.sum()
        gx = self.g(x)
        m = (w * gx).sum()
        return float((w * (gx - m) ** 2).sum())

    def _u_gauss(self, clean, t: np.ndarray) -> np.ndarray:
        mu, sd = clean
        out = np.zeros(t.size, complex)
        for side in (1.0, -1.0):
            R = max(0.0, side * mu) + 12 * sd
            x = side * R * self._w ** 2
            wx = self._wt * 2 * R * self._w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            gx = self.g(x)
            for i0 in range(0, t.size, 200):
                tt = t[i0:i0 + 200, None]
                out[i0:i0 + 200] += (_one_minus_eix(tt * gx[None, :]) * wx[None, :]).sum(1)
        return out

    def _dtheta_env(self, t, x):
        """Upper envelope of |d theta / du| at x = e^u."""
        return t * sum(abs(kj) * a * x ** a for kj, (_, a) in zip(self.k, self.cols))

    def _u_pareto(self, t: float, side: float) -> complex:
        """int_1^inf 3 x^-4 (1 - exp(i t g(side x))) dx, by panels in u = log x plus an IBP tail."""
        top = self.cols[-1][1]
        kt = abs(self.k[-1])
        if kt == 0:
            raise ValueError("top coefficient is zero; the IBP tail needs a dominant top column")
        # end of the panel region: past every stationary point (top term ten times the
        # lower envelope) and oscillating fast enough for one IBP term
        x_osc = (4e3 / (t * kt * top)) ** (1 / top)
        x_dom = max((10 * abs(kj) * a / (kt * top)) ** (1 / (top - a))
                    for kj, (_, a) in zip(self.k[:-1], self.cols[:-1]))
        x_end = max(10.0, x_osc, x_dom)
        u_end = math.log(x_end)
        ub = np.linspace(0.0, u_end, 4001)
        rate = np.maximum(self._dtheta_env(t, np.exp(ub)) / 0.2, 1 / 0.02)
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(ub))])
        npan = int(math.ceil(tau[-1]))
        if npan > 2_000_000:
            raise RuntimeError(f"{npan} panels at t={t:g}; oscillation region too long")
        edges = np.interp(np.linspace(0, tau[-1], npan + 1), tau, ub)
        a, b = edges[:-1, None], edges[1:, None]
        u = 0.5 * (b - a) * _GL_X[None, :] + 0.5 * (a + b)
        wu = (0.5 * (b - a) * _GL_W[None, :]).ravel()
        u = u.ravel()
        x = np.exp(u)
        th = t * self.g(side * x)
        head = (3 * np.exp(-3 * u) * _one_minus_eix(th) * wu).sum()
        # tail: int_X^inf 3x^-4 dx - int_X^inf 3x^-4 e^{i theta} dx, the latter by one IBP term
        X = x_end
        thX = t * self.g(side * X)[0]
        h = 1e-6 * X
        dth = t * (self.g(side * (X + h))[0] - self.g(side * (X - h))[0]) / (2 * h)
        fX = 3 * X ** -4
        tail_osc = 1j * fX * np.exp(1j * thX) / dth
        return complex(head + X ** -3 - tail_osc)

    # ------------------------------------------------------------------ AUC
    def logcf(self, eps_te: tuple[float, float], pplus: float) -> np.ndarray:
        uq = pplus * self.UQp + (1 - pplus) * self.UQm
        if eps_te == (0.0, 0.0):
            uq = np.zeros_like(uq)
        Ap = (1 - eps_te[0]) * self.UGp + eps_te[0] * uq
        Am = (1 - eps_te[1]) * self.UGm + eps_te[1] * uq
        lp, lm = _log1m(Ap), _log1m(Am)
        lp = lp.real + 1j * np.unwrap(lp.imag)
        lm = lm.real + 1j * np.unwrap(lm.imag)
        return lp + np.conj(lm)

    def auc_curve(self, L: np.ndarray, Ns, npts: int = 20000) -> np.ndarray:
        re = interpolate.CubicSpline(self.s, L.real)
        im = interpolate.CubicSpline(self.s, L.imag)
        absL = np.abs(L)
        out = np.empty(len(Ns))
        for i, N in enumerate(Ns):
            lo_idx = np.nonzero(N * absL > 1e-10)[0]
            if lo_idx.size == 0 or lo_idx[0] == 0:
                raise RuntimeError(f"t-grid too short at the low end for N={N:g}")
            s_lo = self.s[lo_idx[0] - 1]
            hi_idx = np.nonzero(N * L.real < -45)[0]
            hi_idx = hi_idx[hi_idx > lo_idx[0]]
            if hi_idx.size == 0:
                raise _HighEndShort(f"t-grid too short at the high end for N={N:g}")
            s_hi = self.s[hi_idx[0]]
            ss = np.linspace(s_lo, s_hi, npts)
            f = np.exp(N * re(ss)) * np.sin(N * im(ss))
            out[i] = 0.5 + integrate.simpson(f, x=ss) / math.pi
        return out


def onset(Ns: np.ndarray, auc: np.ndarray, level: float) -> float:
    """First N past the peak where AUC_N falls to `level` (log-linear interpolation); nan if never."""
    ok = np.isfinite(auc)
    Ns, auc = np.asarray(Ns)[ok], auc[ok]
    if auc.size == 0:
        return float("nan")
    ip = int(np.argmax(auc))
    if auc[ip] <= level:
        return float("nan")
    below = np.nonzero(auc[ip:] <= level)[0]
    if below.size == 0:
        return float("nan")
    j = ip + below[0]
    x0, x1 = math.log(Ns[j - 1]), math.log(Ns[j])
    y0, y1 = auc[j - 1], auc[j]
    return math.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def nstar(cols, K: np.ndarray, eps_te: tuple[float, float], pplus: float) -> float:
    """N*_n of the crossover theorem for the vector K."""
    at = cols[-1][1]
    mu = float(K[:-1] @ test_mean_diff_L(cols, eps_te, pplus))
    kt = abs(K[-1])
    if kt == 0:
        return float("inf")
    if mu <= 0:
        return 1.0
    e = sum(eps_te)
    return (mu / kt) ** (ALPHA / (at - ALPHA)) * e ** (-at / (at - ALPHA))


# --------------------------------------------------------------------------- study blocks

NS_A = (500, 2000, 8000, 32000, 128000)
R_A = 300
NS_B = (50, 200, 1000)
R_B = 40
N_GRID_B = 10 ** np.arange(1.0, 18.0001, 0.05)
GAM_B = ALPHA / 4.0


def clean_windows(cols, N: int, pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window sums of phi over clean test windows (N draws each), for both classes."""
    rng = np.random.default_rng(seed)
    xp = rng.normal(*G_PLUS, (pairs, N)).ravel()
    xm = rng.normal(*G_MINUS, (pairs, N)).ravel()
    Sp = phi(cols, xp).reshape(pairs, N, -1).sum(1)
    Sm = phi(cols, xm).reshape(pairs, N, -1).sum(1)
    return Sp, Sm


def block_a(name: str, astar: float, pplus: float, split: tuple[float, float], seed: int,
            memo: dict) -> dict:
    cols = columns(name, astar)
    aL = cols[-2][1]
    ks = ksub(cols, split, pplus)
    kfull = np.append(ks, 0.0)
    key = ("win", name, astar)
    if key not in memo:
        memo[key] = clean_windows(cols, 10, 20000, seed_of({"win": name, "astar": astar}))
    Sp, Sm = memo[key]
    dS = Sp - Sm
    auc = lambda k: float(np.mean(dS @ k > 0))
    auc_sub = auc(kfull)
    rng = np.random.default_rng(seed)
    out = {"Ksub_lstar": float(ks[-1]), "auc_sub": auc_sub,
           "slope_pred": -(astar - aL) / ALPHA, "same_parity": float(same_parity(cols))}
    med = []
    for n in NS_A:
        Kt = np.empty(R_A); errL = np.empty(R_A); dauc = np.empty(R_A); cond = np.empty(R_A)
        for r in range(R_A):
            xp = draw(rng, n, split[0], G_PLUS, pplus)
            xm = draw(rng, n, split[1], G_MINUS, pplus)
            K, c = fit(cols, xp, xm)
            Kt[r], cond[r] = K[-1], c
            errL[r] = np.linalg.norm(K[:-1] - ks)
            dauc[r] = abs(auc(K) - auc_sub)
        tag = f"n{n}"
        med.append(float(np.median(np.abs(Kt))))
        out[f"medKt_{tag}"] = med[-1]
        out[f"lock_{tag}"] = float(np.mean(np.sign(Kt) == -np.sign(ks[-1])))
        out[f"pos_{tag}"] = float(np.mean(Kt > 0))
        out[f"errKL_{tag}"] = float(np.median(errL))
        out[f"dauc_{tag}"] = float(np.median(dauc))
        out[f"cond_{tag}"] = float(np.median(cond))
    lx, ly = np.log(NS_A[-3:]), np.log(med[-3:])
    out["slope"] = float(np.polyfit(lx, ly, 1)[0])
    out["slope_err"] = out["slope"] - out["slope_pred"]
    b = (NS_A[-1] * sum(split)) ** (1 / ALPHA)
    W = -b ** (astar - aL) * Kt / ks[-1]
    hkey = ("H", name, astar, pplus)
    if hkey not in memo:
        memo[hkey] = h_sample(cols, pplus, 20000,
                              np.random.default_rng(seed_of({"H": name, "astar": astar, "p": pplus})))
    out["ks_H"] = ks_distance(W, memo[hkey])
    out["medW_over_medH"] = float(np.median(W) / np.median(memo[hkey]))
    return out


def block_b_rep(args) -> dict:
    name, n, seed = args
    cols = columns(name, 4.0)
    rng = np.random.default_rng(seed)
    split, pplus = (0.05, 0.05), 0.5
    K, _ = fit(cols, draw(rng, n, split[0], G_PLUS, pplus), draw(rng, n, split[1], G_MINUS, pplus))
    ks = ksub(cols, split, pplus)
    E = CFEngine(cols, K)
    # S1: symmetric test contamination, limit 1/2
    s1 = (0.05, 0.05)
    auc = E.auc(s1, pplus, N_GRID_B)
    n_on = onset(N_GRID_B, auc, 0.75)
    ns1 = nstar(cols, K, s1, pplus)
    # S2: one-sided test contamination
    s2 = (0.10, 0.0)
    ns2 = nstar(cols, K, s2, pplus)
    beta = 1.0 if cols[-1][0] == "e" else (2 * pplus - 1)
    lim = rho(GAM_B, math.copysign(1.0, K[-1]) * beta)
    N2 = 1e6 * ns2
    a2 = float(E.auc(s2, pplus, [N2])[0]) if N2 <= 1e18 else float("nan")
    return {"Kt": float(K[-1]), "lock": float(np.sign(K[-1]) == -np.sign(ks[-1])),
            "peak": float(np.nanmax(auc)), "N_peak": float(N_GRID_B[int(np.nanargmax(auc))]),
            "N_on": n_on, "Nstar": ns1, "auc_end": float(auc[-1]),
            "auc_nan_frac": float(np.mean(np.isnan(auc))),
            "N_first_finite": float(N_GRID_B[np.isfinite(auc)][0]),
            "S2_auc": a2, "S2_lim": lim}


def block_b(name: str, seed: int, workers: int = 8) -> dict:
    import multiprocessing as mp
    ss = np.random.SeedSequence(seed)
    jobs = []
    for n, child in zip(NS_B, ss.spawn(len(NS_B))):
        for c in child.spawn(R_B):
            jobs.append((name, n, int(c.generate_state(1)[0])))
    with mp.get_context("spawn").Pool(workers) as pool:
        reps = pool.map(block_b_rep, jobs)
    lam3 = lambda_mid(GAM_B, 0.75) ** (ALPHA / (4.0 - ALPHA))
    cols = columns(name, 4.0)
    aL = cols[-2][1]
    out = {"slope_pred": (4.0 - aL) / (4.0 - ALPHA), "lambda_m_pow": lam3}
    med_on = []
    for i, n in enumerate(NS_B):
        rs = reps[i * R_B:(i + 1) * R_B]
        on = np.array([r["N_on"] for r in rs])
        ns = np.array([r["Nstar"] for r in rs])
        ok = np.isfinite(on)
        tag = f"n{n}"
        med_on.append(float(np.median(on[ok])) if ok.any() else float("nan"))
        out[f"medNon_{tag}"] = med_on[-1]
        out[f"nan_on_{tag}"] = float(np.mean(~ok))
        out[f"medNstar_{tag}"] = float(np.median(ns))
        out[f"ratio_{tag}"] = float(np.median(on[ok] / (ns[ok] * lam3))) if ok.any() else float("nan")
        out[f"peak_{tag}"] = float(np.median([r["peak"] for r in rs]))
        out[f"aucend_{tag}"] = float(np.median([r["auc_end"] for r in rs]))
        out[f"maxNfirst_{tag}"] = float(max(r["N_first_finite"] for r in rs))
        out[f"medNpeak_{tag}"] = float(np.median([r["N_peak"] for r in rs]))
        out[f"minNpeak_{tag}"] = float(min(r["N_peak"] for r in rs))
        out[f"lock_{tag}"] = float(np.mean([r["lock"] for r in rs]))
        out[f"pos_{tag}"] = float(np.mean([r["Kt"] > 0 for r in rs]))
        s2 = np.array([abs(r["S2_auc"] - r["S2_lim"]) for r in rs])
        out[f"S2err_{tag}"] = float(np.nanmedian(s2)) if np.isfinite(s2).any() else float("nan")
        out[f"S2nan_{tag}"] = float(np.mean(np.isnan(s2)))
    out["slope"] = float(np.polyfit(np.log(NS_B), np.log(med_on), 1)[0])
    out["slope_err"] = out["slope"] - out["slope_pred"]
    return out
