"""Engine for supplement S5 (design and registered criteria: DESIGN_S5.md).

S5 answers two questions raised in review of the manuscript that need computation:
  F. the limit law of Cor. detwindow itself. The two-column, equal-scale design of S3-C
     (alpha_c = 3, a_L = 1, a* = 4), whose training fits are reproduced with their own seeds; for
     each fit the window length N_n = ceil(kappa_0 (c_0 n eps_tr)^{(a*-a_L)/(a*-alpha_c)}) is fixed
     in advance and AUC_{N_n}(K_hat) is computed exactly by the characteristic-function engine of
     run 1 (no window Monte Carlo); its law over fits is compared with the law of
     Psi(-kappa_tilde K^sub_{l*} H);
  G. a one-sided check of (x, |x|^3) at alpha_c in (2, 3) in which the collapse does happen: the
     test split is chosen so that sgn(K_hat_t) beta_te = -1 for a registered share of the fits.

h1v4.py, h1v4_s2.py, h1v4_s3.py and h1v4_s4.py are imported and not modified.
"""
from __future__ import annotations

import math

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T
import h1v4_s4 as F

# --------------------------------------------------------------------------- stable law

def lepage_c(gam: float, beta: float, size: int, rng, terms: int = 4000, chunk: int = 2000) -> np.ndarray:
    """Z_{gam,beta} (Levy tails (1+beta)/2 y^-gam and (1-beta)/2 y^-gam, gam < 1) by the LePage sum
    with its tail beyond `terms` restored: mean beta gam/(1-gam) Gamma_K^(1-1/gam) plus a normal term
    of variance Gamma_K^(1-2/gam)/(2/gam-1). h1v4_s2.lepage drops this tail; its mean is
    beta * 0.13 at gam = 0.73 and beta * 0.95 at gam = 0.83 (terms = 4000), which matters only when
    beta != 0 and gam is close to 1 (found in the S5 design probe, DESIGN_S5.md)."""
    if not 0 < gam < 1:
        raise ValueError("lepage_c needs 0 < gam < 1")
    out = np.empty(size)
    p = (1 + beta) / 2
    for s in range(0, size, chunk):
        b = min(chunk, size - s)
        G = np.cumsum(rng.exponential(1.0, (b, terms)), axis=1)
        xi = np.where(rng.random((b, terms)) < p, 1.0, -1.0)
        GK = G[:, -1]
        z = (xi * G ** (-1 / gam)).sum(1)
        z += beta * gam / (1 - gam) * GK ** (1 - 1 / gam)
        z += np.sqrt(GK ** (1 - 2 / gam) / (2 / gam - 1)) * rng.standard_normal(b)
        out[s:s + b] = z
    return out


def stable_cdf(x: float, gam: float, beta: float) -> float:
    """P(Z_{gam,beta} <= x) by Gil-Pelaez, with log CF -sigma^gam |t|^gam (1 - i beta sgn(t) tan(pi gam/2))
    and sigma^gam = Gamma(1-gam) cos(pi gam/2), the law whose Levy tails total y^-gam. Independent of
    the LePage sums; used by control C18."""
    from scipy import integrate
    c = H.stable_scale_pow(gam)
    tb = beta * math.tan(math.pi * gam / 2)

    def f(t):
        if t == 0:
            return 0.0
        m = c * t ** gam
        return math.exp(-m) * math.sin(m * tb - t * x) / t

    v = 0.0
    edges = [0.0] + [10.0 ** k for k in range(-6, 4)] + [np.inf]
    for a, b in zip(edges[:-1], edges[1:]):
        v += integrate.quad(f, a, b, limit=400, epsabs=1e-12)[0]
    return 0.5 - v / math.pi   # F(x) = 1/2 - (1/pi) int_0^inf Im[e^{-itx} phi(t)] / t dt


# --------------------------------------------------------------------------- block F

F_DICTS = T.RATE_DICTS                     # oo4 = (sgn|x|, sgn|x|^4), oe4 = (sgn|x|, |x|^4)
F_CLEAN = T.RATE_CLEAN                     # N(0.3, 1), N(0, 1): equal scales
F_SPLIT = (0.05, 0.05)                     # training split of S3-C
F_NS = T.RATE_NS                           # the S3-C sample sizes; S5 uses the last three
F_ALPHA = 3.0
F_R = T.R_C                                # 300 fits per n, as S3-C
F_TE = {"sym": (0.05, 0.05), "one": (0.10, 0.0)}
U_GRID = np.round(np.arange(0.01, 0.995, 0.01), 2)   # where the two CDFs are compared


class CFEngineClean(H.CFEngine):
    """H.CFEngine with the clean laws passed in instead of the run-1 constants G_PLUS, G_MINUS.
    The contaminant is the run-1 one (signed Pareto, alpha = 3, c_0 = 1)."""

    def __init__(self, cols, k, cleans, n_gauss: int = 3000, s_min: float = -45 * math.log(10),
                 per_decade: int = 60):
        self.cols, self.k = cols, np.asarray(k, float)
        self.cleans = cleans
        self.g_sd = math.sqrt(0.5 * (self._clean_var(cleans[0]) + self._clean_var(cleans[1])))
        self.pareto, self.per_decade = True, per_decade
        t_max = 60.0 / self.g_sd
        nd = int((math.log10(t_max) - s_min / math.log(10)) * per_decade) + 1
        w, wt = np.polynomial.legendre.leggauss(n_gauss)
        self._w, self._wt = 0.5 * (w + 1), 0.5 * wt
        self.s = np.linspace(s_min, math.log(t_max), nd)
        self.t = np.exp(self.s)
        self.UGp, self.UGm, self.UQp, self.UQm = self._values(self.t)

    def _values(self, t):
        UGp = self._u_gauss(self.cleans[0], t)
        UGm = self._u_gauss(self.cleans[1], t)
        UQp = np.array([self._u_pareto(x, +1.0) for x in t])
        UQm = np.array([self._u_pareto(x, -1.0) for x in t])
        return UGp, UGm, UQp, UQm


def f_seed(name: str) -> int:
    """The seed of the S3-C cell (spec_s3: H.seed_of(params) with params {'block': 'C', 'dict': name})."""
    return H.seed_of({"block": "C", "dict": name})


def f_fits(name: str, n: int) -> np.ndarray:
    """The R_C fits of S3-C at sample size n, same seeds and same draws as h1v4_s3.rate_block."""
    cols = F_DICTS[name]
    cont = T.Pareto(F_ALPHA)
    cleans = (S.Gauss(*F_CLEAN[0]), S.Gauss(*F_CLEAN[1]))
    i = list(F_NS).index(n)
    child = np.random.SeedSequence(f_seed(name)).spawn(len(F_NS))[i]
    out = np.empty((F_R, len(cols)))
    for r, c in enumerate(child.spawn(F_R)):
        rng = np.random.default_rng(int(c.generate_state(1)[0]))
        xp = S.draw(rng, n, F_SPLIT[0], cleans[0], cont)
        xm = S.draw(rng, n, F_SPLIT[1], cleans[1], cont)
        out[r] = H.fit(cols, xp, xm)[0]
    return out


def f_population(name: str) -> dict:
    """K^sub, mu^te_L per split, beta_te per split, gamma, lambda_m (the 0.75 point)."""
    cols = F_DICTS[name]
    cont = T.Pareto(F_ALPHA)
    cleans = (S.Gauss(*F_CLEAN[0]), S.Gauss(*F_CLEAN[1]))
    ks = S.ksub(cols, F_SPLIT, cleans, cont)
    at, aL = cols[-1][1], cols[0][1]
    gam = F_ALPHA / at
    return {"ksub": ks.tolist(), "ksub_lstar": float(ks[-1]), "gamma": gam, "aL": aL, "astar": at,
            "mu_te_L": {k: float(ks @ S.test_mean_diff_L(cols, e, cleans, cont)) for k, e in F_TE.items()},
            "beta_te": {k: S.beta_te(cols, e, cont) for k, e in F_TE.items()},
            "lambda_m": H.lambda_mid(gam, 0.75), "same_parity": bool(H.same_parity(cols))}


def h_draws(name: str, size: int = 20000) -> np.ndarray:
    cols = F_DICTS[name]
    return H.h_sample(cols, 0.5, size, np.random.default_rng(H.seed_of({"H5": name})), campbell=True)


def kappa0_rule(name: str) -> float:
    """Registered rule (DESIGN_S5.md): kappa_0 such that the limit law has its median fit at
    AUC 0.75 under the symmetric split, i.e. kappa_tilde |K^sub_{l*}| median|H| = lambda_m."""
    P = f_population(name)
    mH = float(np.median(np.abs(h_draws(name))))
    a, at = F_ALPHA, P["astar"]
    e = sum(F_TE["sym"])
    # kappa_tilde = kappa_0^{(a*-a)/a} e^{a*/a} / mu  =>  kappa_0 = (lambda mu / (|K| mH e^{a*/a}))^{a/(a*-a)}
    target = P["lambda_m"] * P["mu_te_L"]["sym"] / (abs(P["ksub_lstar"]) * mH * e ** (at / a))
    return float(target ** (a / (at - a)))


def window_length(kappa0: float, n: int, astar: float, aL: float) -> int:
    return int(math.ceil(kappa0 * (n * sum(F_SPLIT)) ** ((astar - aL) / (astar - F_ALPHA))))


def psi_sorted(Zs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Psi(x) = P{1 + x Z > 0} from sorted draws of Z."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    m = Zs.size
    pos, neg = x > 0, x < 0
    out[pos] = 1 - np.searchsorted(Zs, -1 / x[pos], side="right") / m
    out[neg] = np.searchsorted(Zs, 1 / -x[neg], side="left") / m
    out[x == 0] = 1.0
    return out


def f_auc_job(args) -> dict:
    """Exact AUC_{N}(K) under both test splits for one fitted vector K."""
    name, K, N = args
    cols = F_DICTS[name]
    cleans = F_CLEAN
    eng = CFEngineClean(cols, np.asarray(K), cleans)
    out = {}
    for te, e in F_TE.items():
        v = eng.auc(e, 0.5, [N])[0]
        out[te] = float(v)
    return out


def grid_ks(a: np.ndarray, b: np.ndarray, u=U_GRID) -> float:
    """max over u in U_GRID of |F_a(u) - F_b(u)|: the Kolmogorov distance read on a 0.01 grid, so
    that an atom at AUC = 1 in the limit and a value 1 - 1e-9 in a fit are not told apart."""
    a, b = np.sort(a), np.sort(b)
    Fa = np.searchsorted(a, u, side="right") / a.size
    Fb = np.searchsorted(b, u, side="right") / b.size
    return float(np.abs(Fa - Fb).max())


def boot_ci(x: np.ndarray, stat, B: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    v = [stat(x[rng.integers(0, x.size, x.size)]) for _ in range(B)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def f_cell(name: str, n: int, kappa0: float, workers: int = 8) -> dict:
    import multiprocessing as mp
    cols = F_DICTS[name]
    P = f_population(name)
    a, at, aL, gam = F_ALPHA, P["astar"], P["aL"], P["gamma"]
    Kf = f_fits(name, n)
    N = window_length(kappa0, n, at, aL)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(f_auc_job, [(name, K.tolist(), N) for K in Kf])
    Hs = h_draws(name)
    b_n = (n * sum(F_SPLIT)) ** ((at - aL) / a)
    Hn = -b_n * Kf[:, -1] / P["ksub_lstar"]
    out = {"n": n, "N_n": N, "kappa0": kappa0, "ksub_lstar": P["ksub_lstar"], "gamma": gam,
           "medKt": float(np.median(np.abs(Kf[:, -1]))), "pos": float(np.mean(Kf[:, -1] > 0)),
           "ks_Hn_H": H.ks_distance(Hn, Hs), "fit_Kt": Kf[:, -1].tolist(), "fit_KL": Kf[:, 0].tolist()}
    cleans = (S.Gauss(*F_CLEAN[0]), S.Gauss(*F_CLEAN[1]))
    cont = T.Pareto(a)
    for te, e in F_TE.items():
        mu = P["mu_te_L"][te]
        beta = P["beta_te"][te]
        kt = kappa0 ** ((at - a) / a) * sum(e) ** (at / a) / mu
        Z = np.sort(lepage_c(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Zc": gam, "beta": beta}))))
        lim = psi_sorted(Z, -kt * P["ksub_lstar"] * Hs)            # Psi(-kappa_tilde K^sub H)
        auc = np.array([r[te] for r in res])
        # the finite-n conditional limit of Thm crossover, Psi((N_n / N*_n)^{(a*-a)/a} sgn K_t)
        ns = np.array([T.nstar(cols, K, e, cleans, cont) for K in Kf])
        xn = (N / ns) ** ((at - a) / a) * np.sign(Kf[:, -1])
        cond = psi_sorted(Z, xn)
        fin = np.isfinite(auc)
        out[te] = {"beta_te": beta, "mu_te_L": mu, "kappa_tilde": kt, "nan": int((~fin).sum()),
                   "ks_total": grid_ks(auc[fin], lim), "ks_window": grid_ks(auc[fin], cond[fin]),
                   "ks_training": grid_ks(cond, lim),
                   "mean_auc": float(auc[fin].mean()), "mean_lim": float(lim.mean()),
                   "mean_diff": float(auc[fin].mean() - lim.mean()),
                   "mean_auc_ci": boot_ci(auc[fin], np.mean, seed=H.seed_of({"ci": name, "n": n, "te": te})),
                   "q_auc": [float(v) for v in np.quantile(auc[fin], (0.1, 0.25, 0.5, 0.75, 0.9))],
                   "q_lim": [float(v) for v in np.quantile(lim, (0.1, 0.25, 0.5, 0.75, 0.9))],
                   "max_abs_auc_minus_cond": float(np.max(np.abs(auc[fin] - cond[fin]))),
                   "fit_auc": auc.tolist(), "fit_cond": cond.tolist()}
    return out


# --------------------------------------------------------------------------- block G

G_COLS = (("o", 1.0), ("e", 3.0))          # (x, |x|^3), a_L = 1, a* = 3
G_ASTAR = 3.0
G_R = 200
MIRROR = ((0.15, 1.0), (-0.15, 1.0))       # mirror-image clean laws: every even moment is equal


def g_cleans(src: str):
    if src == "mirror":
        g = (S.Gauss(*MIRROR[0]), S.Gauss(*MIRROR[1]))
        return g, g
    return F.cleans_of(src)


def g_rep(args) -> dict:
    """One training replication of a block-G cell. Same steps as h1v4_s4.crossover_rep, with the
    clean laws of g_cleans; adds the clean drift of the top column for the N*_eff prediction."""
    (alpha, n, eps_tr, eps_te, src, seed) = args
    cols = [tuple(c) for c in G_COLS]
    cont = T.Pareto(alpha)
    tr, te = g_cleans(src)
    rng = np.random.default_rng(seed)
    xp = S.draw(rng, n, eps_tr[0], tr[0], cont)
    xm = S.draw(rng, n, eps_tr[1], tr[1], cont)
    K, KW, KS, thr, cond = S.fit_all(cols, xp, xm)
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(KW)) and np.all(np.isfinite(KS))):
        return {"failed": 1.0}
    a = S.window_auc(rng, cols, K, KW, KS, thr, eps_te, te, cont)
    ks = S.ksub(cols, eps_tr, tr, cont)
    w10, s10 = S.clean_auc10(rng, cols, KW, KS, thr, te)
    return {"failed": 0.0, "Kt": float(K[-1]), "K": K.tolist(), "cond": cond,
            "lock": float(np.sign(K[-1]) == -np.sign(ks[-1])),
            "Nstar": T.nstar(cols, K, eps_te, te, cont), "N_on": H.onset(S.N_GRID, a["K"], 0.75),
            "peak": float(np.max(a["K"])), "auc_end": float(a["K"][-1]),
            "aucW_end": float(a["KW"][-1]), "aucS_end": float(a["KS"][-1]),
            "aucW10_clean": w10, "aucS10_clean": s10,
            "curve_K": a["K"].tolist(), "curve_KW": a["KW"].tolist(), "curve_KS": a["KS"].tolist()}


def g_cell(alpha: float, n: int, eps_te, src: str, seed: int, eps_tr=(0.05, 0.05), workers: int = 8) -> dict:
    import multiprocessing as mp
    cols = [tuple(c) for c in G_COLS]
    seeds = [int(c.generate_state(1)[0]) for c in np.random.SeedSequence(seed).spawn(G_R)]
    with mp.get_context("spawn").Pool(workers) as pool:
        reps = pool.map(g_rep, [(alpha, n, eps_tr, eps_te, src, s) for s in seeds])
    tr, te = g_cleans(src)
    cont = T.Pareto(alpha)
    out = F.summarize(reps, cols, G_ASTAR, alpha, eps_tr, eps_te, cont, tr, te)
    ok = [r for r in reps if r["failed"] == 0.0]
    beta = out["beta_te"]
    sg = np.array([math.copysign(1.0, r["Kt"]) for r in ok])
    coll = sg * beta < 0                                   # fits whose limit past N* is an inversion
    end = np.array(out["fit_end"])
    on = np.array(out["fit_Non"])
    # F.summarize predicts with the uncompensated LePage sum; keep it for the record, predict again
    # with lepage_c, and recompute the summary keys that depend on the prediction
    Zc = np.sort(lepage_c(alpha / G_ASTAR, beta, 200_000,
                          np.random.default_rng(H.seed_of({"Zc": alpha / G_ASTAR, "beta": beta}))))
    lam = lambda c: c ** ((G_ASTAR - alpha) / alpha)
    ns_fit = np.array(out["fit_Nstar"])
    pred = psi_sorted(Zc, lam(S.N_GRID[-1] / ns_fit) * sg)
    out.update({"fit_pred_uncomp": out["fit_pred"], "predErr_uncomp": out["predErr"],
                "fit_pred": pred.tolist(), "medPred": float(np.median(pred)),
                "predErr": float(np.median(np.abs(end - pred))), "errQ": F._q(np.abs(end - pred))})
    # N*_eff: the admissible drift plus the clean drift of the top column (diag_s4_clean_drift.py)
    par, at = cols[-1]
    dt = (1 - eps_te[0]) * te[0].moment(par, at) - (1 - eps_te[1]) * te[1].moment(par, at)
    dL = S.test_mean_diff_L(cols, eps_te, te, cont)
    e = cont.c0 * sum(eps_te)
    ne = []
    for r in ok:
        K = np.asarray(r["K"])
        mu = float(K[:-1] @ dL) + K[-1] * dt
        ne.append((mu / abs(K[-1])) ** (alpha / (at - alpha)) * e ** (-at / (at - alpha)) if mu > 0 else 1.0)
    ne = np.array(ne)
    pe = psi_sorted(Zc, lam(S.N_GRID[-1] / ne) * sg)
    err, err_eff = np.abs(end - pred), np.abs(end - pe)
    sub = lambda x, m: x[m] if m.any() else np.array([np.nan])
    seed_ci = H.seed_of({"ci_g": seed})
    out.update({
        "clean_top_diff": float(dt), "share_collapse": float(coll.mean()), "n_collapse": int(coll.sum()),
        "fit_Nstar_eff": ne.tolist(), "fit_pred_eff": pe.tolist(),
        "coll_medAucEnd": float(np.median(sub(end, coll))),
        "coll_medPred": float(np.median(sub(pred, coll))),
        "coll_predErr": float(np.median(sub(err, coll))),
        "coll_predErr_eff": float(np.median(sub(err_eff, coll))),
        "coll_errQ": F._q(sub(err, coll)), "coll_errQ_eff": F._q(sub(err_eff, coll)),
        "coll_onFinite": float(np.mean(np.isfinite(sub(on, coll)))),
        "coll_medNon": float(np.nanmedian(sub(on, coll))) if np.isfinite(sub(on, coll)).any() else float("nan"),
        "coll_medNstar": float(np.median(sub(np.array(out["fit_Nstar"]), coll))),
        "coll_medNstar_eff": float(np.median(sub(ne, coll))),
        "coll_medAucWEnd": float(np.median(sub(np.array([r["aucW_end"] for r in ok]), coll))),
        "coll_medPeak": float(np.median(sub(np.array([r["peak"] for r in ok]), coll))),
        "coll_medDrop": float(np.median(sub(np.array([r["peak"] - r["auc_end"] for r in ok]), coll))),
        "coll_predErr_ci": boot_ci(sub(err, coll), np.median, seed=seed_ci) if coll.sum() > 1 else [math.nan] * 2,
        "coll_predErr_eff_ci": boot_ci(sub(err_eff, coll), np.median, seed=seed_ci + 1) if coll.sum() > 1 else [math.nan] * 2,
        "coll_medAucEnd_ci": boot_ci(sub(end, coll), np.median, seed=seed_ci + 2) if coll.sum() > 1 else [math.nan] * 2,
        "noncoll_medAucEnd": float(np.median(sub(end, ~coll))),
        "predErr_eff": float(np.median(err_eff)),
    })
    return out
