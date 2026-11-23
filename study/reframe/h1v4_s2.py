"""Engine for supplement S2 of the stage-6 study (design and registered criteria: DESIGN_S2.md).

Extends h1v4 in three directions without changing it:
- laws: a clean law is a Gaussian or a finite pool of real values; the contaminant is the
  signed Pareto of h1v4 (c0 = 1) or Student t3 (c0 = 2.2053);
- window AUC by Monte Carlo on prefix sums of M window pairs of length N_max, for the
  unclipped, the winsorized and the fitted sub-dictionary vector on the same windows;
- the conditional prediction of the crossover theorem by LePage sampling of Z_{gamma,beta}.
"""
from __future__ import annotations

import functools
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from scipy import special

import h1v4 as H

ALPHA = H.ALPHA
T3_C0 = 2 * 3 * special.gamma(2.0) / (math.sqrt(3 * math.pi) * special.gamma(1.5))
M_PAIRS = 500
N_GRID = np.unique(np.round(10 ** np.arange(0.5, 4.5001, 0.125)).astype(int))
R_X = 40
WINSOR_Q = (0.005, 0.995)


# --------------------------------------------------------------------------- laws

class Gauss:
    def __init__(self, mu: float, sd: float):
        self.mu, self.sd = mu, sd

    def sample(self, rng, size):
        return rng.normal(self.mu, self.sd, size)

    def moment(self, parity: str, q: float) -> float:
        return H._gauss_moment(parity, q, self.mu, self.sd)


class Pool:
    """A finite pool of real values; i.i.d. draws with replacement; moments are pool means."""

    def __init__(self, values: np.ndarray):
        self.v = np.asarray(values, float)

    def sample(self, rng, size):
        return self.v[rng.integers(0, self.v.size, size)]

    def moment(self, parity: str, q: float) -> float:
        a = np.abs(self.v) ** q
        return float(a.mean() if parity == "e" else (np.sign(self.v) * a).mean())


class Contaminant:
    def __init__(self, kind: str, pplus: float = 0.5):
        if kind not in ("pareto", "t3"):
            raise ValueError(kind)
        if kind == "t3" and pplus != 0.5:
            raise ValueError("t3 is symmetric: pplus must be 0.5")
        self.kind, self.pplus = kind, pplus
        self.c0 = 1.0 if kind == "pareto" else T3_C0

    def sample(self, rng, size):
        if self.kind == "t3":
            return rng.standard_t(3.0, size)
        mag = (1.0 - rng.random(size)) ** (-1.0 / ALPHA)
        return np.where(rng.random(size) < self.pplus, mag, -mag)

    def moment(self, parity: str, q: float) -> float:
        if q >= ALPHA:
            raise ValueError(f"moment of order {q} is infinite")
        if self.kind == "pareto":
            return H._pareto_moment(parity, q, self.pplus)
        if parity == "o":
            return 0.0
        return float(3 ** (q / 2) * special.gamma((q + 1) / 2) * special.gamma((3 - q) / 2)
                     / (math.sqrt(math.pi) * special.gamma(1.5)))


def draw(rng, size, eps: float, clean, cont: Contaminant) -> np.ndarray:
    """i.i.d. draws from (1 - eps) clean + eps cont, contaminant positions random."""
    x = clean.sample(rng, size)
    if eps > 0:
        m = rng.random(size) < eps
        k = int(m.sum())
        if k:
            x[m] = cont.sample(rng, k)
    return x


def _mix(parity, q, eps, clean, cont):
    return (1 - eps) * clean.moment(parity, q) + eps * cont.moment(parity, q)


def pop_mean(cols, eps, clean, cont):
    return np.array([_mix(p, a, eps, clean, cont) for p, a in cols])


def pop_cov(cols, eps, clean, cont):
    m = pop_mean(cols, eps, clean, cont)
    s = len(cols)
    C = np.empty((s, s))
    for i in range(s):
        for j in range(s):
            C[i, j] = _mix(*H._prod(cols[i], cols[j]), eps, clean, cont) - m[i] * m[j]
    return C


def ksub(cols, eps_tr, cleans, cont):
    L = cols[:-1]
    F = pop_cov(L, eps_tr[0], cleans[0], cont) + pop_cov(L, eps_tr[1], cleans[1], cont)
    Y = pop_mean(L, eps_tr[0], cleans[0], cont) - pop_mean(L, eps_tr[1], cleans[1], cont)
    return np.linalg.solve(F, Y)


def test_mean_diff_L(cols, eps_te, cleans, cont):
    L = cols[:-1]
    return pop_mean(L, eps_te[0], cleans[0], cont) - pop_mean(L, eps_te[1], cleans[1], cont)


def nstar(cols, K, eps_te, cleans, cont) -> float:
    """N*_n of the crossover theorem with the contaminant's c0."""
    at = cols[-1][1]
    mu = float(K[:-1] @ test_mean_diff_L(cols, eps_te, cleans, cont))
    kt = abs(K[-1])
    if kt == 0:
        return float("inf")
    if mu <= 0:
        return 1.0
    e = cont.c0 * sum(eps_te)
    return (mu / kt) ** (ALPHA / (at - ALPHA)) * e ** (-at / (at - ALPHA))


def beta_te(cols, eps_te, cont) -> float:
    r, l = (1.0, 0.0) if cols[-1][0] == "e" else (cont.pplus, 1 - cont.pplus)
    return (eps_te[0] - eps_te[1]) * (r - l) / sum(eps_te)


# --------------------------------------------------------------------------- fits

def thresholds(xp, xm):
    x = np.concatenate([xp, xm])
    return tuple(float(v) for v in np.quantile(x, WINSOR_Q))


def fit_all(cols, xp, xm):
    """Unclipped K, winsorized K^W (input clipped at training quantiles), fitted sub-dictionary."""
    K, cond = H.fit(cols, xp, xm)
    thr = thresholds(xp, xm)
    KW, _ = H.fit(cols, np.clip(xp, *thr), np.clip(xm, *thr))
    KS, _ = H.fit(cols[:-1], xp, xm)
    return K, KW, KS, thr, cond


# --------------------------------------------------------------------------- window MC

def window_auc(rng, cols, K, KW, KS, thr, eps_te, cleans, cont,
               Ns=N_GRID, M=M_PAIRS, chunk=20) -> dict:
    """AUC_N of the three vectors on the same M window pairs (prefix sums up to max Ns)."""
    Ns = np.asarray(Ns)
    nmax = int(Ns.max())
    idx = Ns - 1
    wins = {k: np.zeros(Ns.size) for k in ("K", "KW", "KS")}
    for s in range(0, M, chunk):
        b = min(chunk, M - s)
        xp = draw(rng, (b, nmax), eps_te[0], cleans[0], cont).ravel()
        xm = draw(rng, (b, nmax), eps_te[1], cleans[1], cont).ravel()
        for key, kk, cc, clip in (("K", K, cols, False), ("KW", KW, cols, True),
                                  ("KS", KS, cols[:-1], False)):
            gp = H.phi(cc, np.clip(xp, *thr) if clip else xp) @ kk
            gm = H.phi(cc, np.clip(xm, *thr) if clip else xm) @ kk
            d = np.cumsum(gp.reshape(b, nmax) - gm.reshape(b, nmax), axis=1)[:, idx]
            wins[key] += (d > 0).sum(0) + 0.5 * (d == 0).sum(0)
    return {k: v / M for k, v in wins.items()}


def clean_auc10(rng, cols, KW, KS, thr, cleans, pairs=20000) -> tuple[float, float]:
    xp = cleans[0].sample(rng, (pairs, 10)).ravel()
    xm = cleans[1].sample(rng, (pairs, 10)).ravel()
    out = []
    for kk, cc, clip in ((KW, cols, True), (KS, cols[:-1], False)):
        gp = (H.phi(cc, np.clip(xp, *thr) if clip else xp) @ kk).reshape(pairs, 10).sum(1)
        gm = (H.phi(cc, np.clip(xm, *thr) if clip else xm) @ kk).reshape(pairs, 10).sum(1)
        out.append(float(np.mean(gp > gm) + 0.5 * np.mean(gp == gm)))
    return out[0], out[1]


# --------------------------------------------------------------------------- prediction

def lepage(gam: float, beta: float, size: int, rng, terms: int = 4000, chunk: int = 2000) -> np.ndarray:
    """Z_{gam,beta} with Levy tails (1+beta)/2 y^-gam, (1-beta)/2 y^-gam: sum xi_k Gamma_k^(-1/gam)."""
    out = np.empty(size)
    p = (1 + beta) / 2
    for s in range(0, size, chunk):
        b = min(chunk, size - s)
        G = np.cumsum(rng.exponential(1.0, (b, terms)), axis=1)
        xi = np.where(rng.random((b, terms)) < p, 1.0, -1.0)
        out[s:s + b] = (xi * G ** (-1 / gam)).sum(1)
    return out


def predict(Z: np.ndarray, c: float, sign: float, astar: float) -> float:
    """P{1 + c^((a*-alpha)/alpha) * sign * Z > 0} from LePage draws of Z."""
    lam = c ** ((astar - ALPHA) / ALPHA)
    return float(np.mean(1 + lam * sign * Z > 0))


# --------------------------------------------------------------------------- RflyMAD pools

# Per-flight RflyMAD series (no-fault real flights); MPFD_RFLY_CACHE overrides the location.
RFLY_CACHE = Path(os.environ.get("MPFD_RFLY_CACHE",
                                 Path(__file__).resolve().parent / "data" / "ext_cache_f.npz"))


@functools.lru_cache(maxsize=2)
def rfly_pools(seed: int) -> dict:
    """Hover / cruise pools of airborne steady-altitude vertical acceleration, split by flight."""
    fl = np.load(RFLY_CACHE, allow_pickle=True)["flights"]
    per = []
    for d in fl:
        t, a = np.asarray(d["acc_t"], float), np.asarray(d["a_up"], float)
        tg = np.asarray(d["tg"], float)
        if t.size < 100 or tg.size < 2:
            continue
        v = np.interp(t, tg, np.asarray(d["vxy"], float))
        vz = np.interp(t, tg, np.asarray(d["vz"], float))
        alt = np.asarray(d["alt_gps"], float)
        z = np.interp(t, tg, alt - np.min(alt))
        air = (z > 3.0) & (np.abs(vz) < 0.5)
        if not air.any():
            continue
        a = a - np.median(a[air])
        lo, hi = a[air & (v < 1.0)], a[air & (v > 4.0)]
        if lo.size >= 500 and hi.size >= 500:
            per.append((hi, lo))
    perm = np.random.default_rng(seed).permutation(len(per))
    half = len(per) // 2
    tr, te = perm[:half], perm[half:]
    cat = lambda ids, j: np.concatenate([per[i][j] for i in ids])
    s = np.std(np.concatenate([cat(tr, 0), cat(tr, 1)]))
    sha = hashlib.sha256(RFLY_CACHE.read_bytes()).hexdigest()
    return {"train": (Pool(cat(tr, 0) / s), Pool(cat(tr, 1) / s)),
            "test": (Pool(cat(te, 0) / s), Pool(cat(te, 1) / s)),
            "n_flights": len(per), "train_ids": [int(i) for i in tr], "scale": float(s),
            "cache_sha256": sha}


# --------------------------------------------------------------------------- cells

def crossover_rep(args) -> dict:
    """One training replication of a crossover cell; window MC for its test contamination."""
    (name, astar, n, eps_te, cont_kind, src, seed) = args
    cols = H.columns(name, astar)
    cont = Contaminant(cont_kind)
    rng = np.random.default_rng(seed)
    if src == "rfly":
        P = rfly_pools(H.seed_of({"rfly_split": 1}))
        tr, te = P["train"], P["test"]
    else:
        tr = te = (Gauss(*H.G_PLUS), Gauss(*H.G_MINUS))
    split = (0.05, 0.05)
    xp, xm = draw(rng, n, split[0], tr[0], cont), draw(rng, n, split[1], tr[1], cont)
    K, KW, KS, thr, cond = fit_all(cols, xp, xm)
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(KW)) and np.all(np.isfinite(KS))):
        return {"failed": 1.0}
    ks = ksub(cols, split, tr, cont)
    ns = nstar(cols, K, eps_te, te, cont)
    a = window_auc(rng, cols, K, KW, KS, thr, eps_te, te, cont)
    w10, s10 = clean_auc10(rng, cols, KW, KS, thr, te)
    return {"failed": 0.0, "Kt": float(K[-1]), "lock": float(np.sign(K[-1]) == -np.sign(ks[-1])),
            "cond": cond, "Nstar": ns, "N_on": H.onset(N_GRID, a["K"], 0.75),
            "peak": float(np.max(a["K"])), "auc_end": float(a["K"][-1]),
            "aucW_end": float(a["KW"][-1]), "aucS_end": float(a["KS"][-1]),
            "aucW10_clean": w10, "aucS10_clean": s10,
            "curve_K": a["K"].tolist(), "curve_KW": a["KW"].tolist(), "curve_KS": a["KS"].tolist()}


def crossover_cell(name, astar, n, eps_te, cont_kind, src, seed, workers=8) -> dict:
    import multiprocessing as mp
    jobs = [(name, astar, n, eps_te, cont_kind, src, int(c.generate_state(1)[0]))
            for c in np.random.SeedSequence(seed).spawn(R_X)]
    with mp.get_context("spawn").Pool(workers) as pool:
        reps = pool.map(crossover_rep, jobs)
    ok = [r for r in reps if r["failed"] == 0.0]
    cols = H.columns(name, astar)
    cont = Contaminant(cont_kind)
    gam = ALPHA / astar
    lam_pow = H.lambda_mid(gam, 0.75) ** (ALPHA / (astar - ALPHA))
    beta = beta_te(cols, eps_te, cont)
    Z = lepage(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Z": gam, "beta": beta})))
    on = np.array([r["N_on"] for r in ok])
    ns = np.array([r["Nstar"] for r in ok])
    fin = np.isfinite(on)
    pred = np.array([predict(Z, N_GRID[-1] / r["Nstar"], math.copysign(1.0, r["Kt"]), astar)
                     for r in ok])
    end = np.array([r["auc_end"] for r in ok])
    med = lambda key: float(np.median([r[key] for r in ok]))
    out = {"failed": float(len(reps) - len(ok)), "beta_te": beta, "lambda_m_pow": lam_pow,
           "medPeak": med("peak"), "onFinite": float(fin.mean()),
           "medNon": float(np.median(on[fin])) if fin.any() else float("nan"),
           "medNstar": float(np.median(ns)),
           "ratio": float(np.median(on[fin] / (lam_pow * ns[fin]))) if fin.any() else float("nan"),
           "medAucEnd": float(np.median(end)), "predErr": float(np.median(np.abs(end - pred))),
           "medAucWEnd": med("aucW_end"), "medAucSEnd": med("aucS_end"),
           "medCostW10": float(np.median([r["aucW10_clean"] - r["aucS10_clean"] for r in ok])),
           "lock": float(np.mean([r["lock"] for r in ok])),
           "pos": float(np.mean([r["Kt"] > 0 for r in ok])), "medCond": med("cond"),
           "N_grid": N_GRID.tolist(),
           "medCurveK": np.median([r["curve_K"] for r in ok], axis=0).tolist(),
           "medCurveKW": np.median([r["curve_KW"] for r in ok], axis=0).tolist(),
           "medCurveKS": np.median([r["curve_KS"] for r in ok], axis=0).tolist()}
    if src == "rfly":
        P = rfly_pools(H.seed_of({"rfly_split": 1}))
        out.update({"n_flights": float(P["n_flights"]), "cache_sha256": P["cache_sha256"]})
    return out


def train_block_t3(name: str, seed: int, ns=(2000, 32000, 128000), R=100, memo=None) -> dict:
    """Block-A analogue with a t3 contaminant (a* = 4, balanced split, p+ = 1/2)."""
    cols = H.columns(name, 4.0)
    cont = Contaminant("t3")
    cl = (Gauss(*H.G_PLUS), Gauss(*H.G_MINUS))
    split = (0.05, 0.05)
    ks = ksub(cols, split, cl, cont)
    Sp, Sm = H.clean_windows(cols, 10, 20000, H.seed_of({"win": name, "astar": 4.0}))
    dS = Sp - Sm
    auc = lambda k: float(np.mean(dS @ k > 0))
    auc_sub = auc(np.append(ks, 0.0))
    rng = np.random.default_rng(seed)
    out = {"Ksub_lstar": float(ks[-1]), "auc_sub": auc_sub, "same_parity": float(H.same_parity(cols))}
    for n in ns:
        Kt, dauc = np.empty(R), np.empty(R)
        for r in range(R):
            K, _ = H.fit(cols, draw(rng, n, split[0], cl[0], cont), draw(rng, n, split[1], cl[1], cont))
            Kt[r], dauc[r] = K[-1], abs(auc(K) - auc_sub)
        out[f"lock_n{n}"] = float(np.mean(np.sign(Kt) == -np.sign(ks[-1])))
        out[f"pos_n{n}"] = float(np.mean(Kt > 0))
        out[f"dauc_n{n}"] = float(np.median(dauc))
    return out
