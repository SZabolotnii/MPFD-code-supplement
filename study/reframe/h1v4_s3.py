"""Engine for supplement S3 (design and registered criteria: DESIGN_S3.md).

S3 asks three things the first two runs left open:
  A. heavier contaminants, alpha_c in {1.2, 1.5}, with cubic and quartic top features and
     realistic contamination (5-10 %);
  B. interior test skewness beta_te and the reverse one-sided split, at alpha_c = 3;
  C. the training-size rate and the limit law of H in a two-column dictionary with equal class
     scales, where the remainders of the finite-n remark are smallest.

The run-1 and run-2 engines (h1v4.py, h1v4_s2.py) fix alpha_c = 3 as a module constant and stay
untouched; this module passes alpha_c explicitly and reuses their alpha-free parts.
"""
from __future__ import annotations

import math

import numpy as np

import h1v4 as H
import h1v4_s2 as S

N_GRID = S.N_GRID
R_X = S.R_X
G_CLEAN = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))


class Pareto:
    """Signed Pareto contaminant of index alpha: P(|C| > x) = x^-alpha for x >= 1, c0 = 1."""

    kind = "pareto"

    def __init__(self, alpha: float, pplus: float = 0.5):
        self.alpha, self.pplus, self.c0 = alpha, pplus, 1.0

    def sample(self, rng, size):
        mag = (1.0 - rng.random(size)) ** (-1.0 / self.alpha)
        return np.where(rng.random(size) < self.pplus, mag, -mag)

    def moment(self, parity: str, q: float) -> float:
        if q >= self.alpha:
            raise ValueError(f"moment of order {q} is infinite")
        m = self.alpha / (self.alpha - q)
        return m if parity == "e" else (2 * self.pplus - 1) * m


def columns(name: str, astar: float, alpha: float) -> list[tuple[str, float]]:
    """The run-1 dictionary layout with the lower exponents scaled by alpha/3, so every lower
    column keeps its ratio to the tail index; the top exponent is a* itself."""
    return [(p, astar if a is None else a * alpha / 3.0) for p, a in H.DICTS[name]]


def nstar(cols, K, eps_te, cleans, cont) -> float:
    at, al = cols[-1][1], cont.alpha
    mu = float(K[:-1] @ S.test_mean_diff_L(cols, eps_te, cleans, cont))
    kt = abs(K[-1])
    if kt == 0:
        return float("inf")
    if mu <= 0:
        return 1.0
    e = cont.c0 * sum(eps_te)
    return (mu / kt) ** (al / (at - al)) * e ** (-at / (at - al))


def predict(Z, c, sign, astar, alpha) -> float:
    lam = c ** ((astar - alpha) / alpha)
    return float(np.mean(1 + lam * sign * Z > 0))


def crossover_rep(args) -> dict:
    """One training replication: fit, N*, window MC on the same windows for K, K^W, K^sub."""
    (name, astar, alpha, n, eps_tr, eps_te, seed) = args
    cols = columns(name, astar, alpha)
    cont = Pareto(alpha)
    rng = np.random.default_rng(seed)
    xp = S.draw(rng, n, eps_tr[0], G_CLEAN[0], cont)
    xm = S.draw(rng, n, eps_tr[1], G_CLEAN[1], cont)
    K, KW, KS, thr, cond = S.fit_all(cols, xp, xm)
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(KW)) and np.all(np.isfinite(KS))):
        return {"failed": 1.0}
    ks = S.ksub(cols, eps_tr, G_CLEAN, cont)
    ns = nstar(cols, K, eps_te, G_CLEAN, cont)
    a = S.window_auc(rng, cols, K, KW, KS, thr, eps_te, G_CLEAN, cont)
    w10, s10 = S.clean_auc10(rng, cols, KW, KS, thr, G_CLEAN)
    return {"failed": 0.0, "Kt": float(K[-1]), "lock": float(np.sign(K[-1]) == -np.sign(ks[-1])),
            "cond": cond, "Nstar": ns, "N_on": H.onset(N_GRID, a["K"], 0.75),
            "peak": float(np.max(a["K"])), "auc_end": float(a["K"][-1]),
            "aucW_end": float(a["KW"][-1]), "aucS_end": float(a["KS"][-1]),
            "aucW10_clean": w10, "aucS10_clean": s10,
            "curve_K": a["K"].tolist(), "curve_KW": a["KW"].tolist(), "curve_KS": a["KS"].tolist()}


def crossover_cell(name, astar, alpha, n, eps_tr, eps_te, seed, workers=8) -> dict:
    import multiprocessing as mp
    jobs = [(name, astar, alpha, n, eps_tr, eps_te, int(c.generate_state(1)[0]))
            for c in np.random.SeedSequence(seed).spawn(R_X)]
    with mp.get_context("spawn").Pool(workers) as pool:
        reps = pool.map(crossover_rep, jobs)
    ok = [r for r in reps if r["failed"] == 0.0]
    cols = columns(name, astar, alpha)
    cont = Pareto(alpha)
    gam = alpha / astar
    lam_pow = H.lambda_mid(gam, 0.75) ** (alpha / (astar - alpha))
    beta = S.beta_te(cols, eps_te, cont)
    Z = S.lepage(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Z": gam, "beta": beta})))
    on = np.array([r["N_on"] for r in ok])
    ns = np.array([r["Nstar"] for r in ok])
    fin = np.isfinite(on)
    pred = np.array([predict(Z, N_GRID[-1] / r["Nstar"], math.copysign(1.0, r["Kt"]), astar, alpha)
                     for r in ok])
    end = np.array([r["auc_end"] for r in ok])
    ksub = S.ksub(cols, eps_tr, G_CLEAN, cont)
    mu_te = float(ksub @ S.test_mean_diff_L(cols, eps_te, G_CLEAN, cont))
    med = lambda key: float(np.median([r[key] for r in ok]))
    return {"failed": float(len(reps) - len(ok)), "beta_te": beta, "lambda_m_pow": lam_pow,
            "ksub_lstar": float(ksub[-1]), "mu_te_L": mu_te,
            "medPeak": med("peak"), "onFinite": float(fin.mean()),
            "medNon": float(np.median(on[fin])) if fin.any() else float("nan"),
            "medNstar": float(np.median(ns)),
            "ratio": float(np.median(on[fin] / (lam_pow * ns[fin]))) if fin.any() else float("nan"),
            "medAucEnd": float(np.median(end)), "predErr": float(np.median(np.abs(end - pred))),
            "medPred": float(np.median(pred)),
            "medAucWEnd": med("aucW_end"), "medAucSEnd": med("aucS_end"),
            "medCostW10": float(np.median([r["aucW10_clean"] - r["aucS10_clean"] for r in ok])),
            "lock": float(np.mean([r["lock"] for r in ok])),
            "pos": float(np.mean([r["Kt"] > 0 for r in ok])), "medCond": med("cond"),
            "N_grid": N_GRID.tolist(),
            "medCurveK": np.median([r["curve_K"] for r in ok], axis=0).tolist(),
            "medCurveKW": np.median([r["curve_KW"] for r in ok], axis=0).tolist(),
            "medCurveKS": np.median([r["curve_KS"] for r in ok], axis=0).tolist()}


# --------------------------------------------------------------------------- block C: rate

RATE_DICTS = {"oo4": [("o", 1.0), ("o", 4.0)], "oe4": [("o", 1.0), ("e", 4.0)]}
RATE_CLEAN = ((0.3, 1.0), (0.0, 1.0))   # equal class scales: no even clean difference
RATE_NS = (2000, 8000, 32000, 128000, 512000)
R_C = 300


def rate_block(name: str, seed: int, split=(0.05, 0.05), ns=RATE_NS, R=R_C) -> dict:
    """Training limits in a two-column dictionary at alpha_c = 3 with equal class scales."""
    cols = RATE_DICTS[name]
    aL, at = cols[0][1], cols[1][1]
    cont = Pareto(3.0)
    cleans = (S.Gauss(*RATE_CLEAN[0]), S.Gauss(*RATE_CLEAN[1]))
    ks = S.ksub(cols, split, cleans, cont)
    out = {"ksub_lstar": float(ks[-1]), "slope_pred": -(at - aL) / 3.0,
           "same_parity": float(H.same_parity(cols))}
    med, last = [], None
    for n, child in zip(ns, np.random.SeedSequence(seed).spawn(len(ns))):
        Kt = np.empty(R)
        for r, c in enumerate(child.spawn(R)):
            rng = np.random.default_rng(int(c.generate_state(1)[0]))
            xp = S.draw(rng, n, split[0], cleans[0], cont)
            xm = S.draw(rng, n, split[1], cleans[1], cont)
            K, _ = H.fit(cols, xp, xm)
            Kt[r] = K[-1]
        med.append(float(np.median(np.abs(Kt))))
        out[f"medKt_n{n}"] = med[-1]
        out[f"lock_n{n}"] = float(np.mean(np.sign(Kt) == -np.sign(ks[-1])))
        out[f"pos_n{n}"] = float(np.mean(Kt > 0))
        last = Kt
    out["slope"] = float(np.polyfit(np.log(ns[-3:]), np.log(med[-3:]), 1)[0])
    out["slope_err"] = out["slope"] - out["slope_pred"]
    b = (ns[-1] * sum(split)) ** (1 / 3.0)
    W = -b ** (at - aL) * last / ks[-1]
    hs = H.h_sample(cols, 0.5, 20000, np.random.default_rng(H.seed_of({"H3": name})), campbell=True)
    out["ks_H"] = H.ks_distance(W, hs)
    out["medW_over_medH"] = float(np.median(W) / np.median(hs))
    return out
