"""Engine for supplement S4 (design and registered criteria: DESIGN_S4.md).

S4 answers the items left open after the fourth manuscript review:
  D. per-fit dispersion of the onset ratio and of the prediction error: every crossover cell of
     runs 2 and 3 is re-run with its own seeds through the unchanged replication functions of
     h1v4_s2 / h1v4_s3, so the medians must reproduce the stored ones exactly, and the per-fit
     values are kept this time; the same fits give the fixed-window check of Cor. detwindow
     (mean AUC_N against the mean conditional prediction at every grid N >= FW_MIN_N);
  A. the two-column dictionaries (x, x^3) and (x, |x|^3) at alpha_c in (2, 3), inside the exact
     theory, and (x, x^2, x^3), which has two inadmissible columns and is reported only;
  B. a locked interior-beta cell (ee, alpha_c = 1.2, a* = 4, n = 1000);
  C. the RflyMAD clean pools in the run-3 regime (alpha_c in {1.2, 1.5}, a* in {3, 4});
  E. the reverse one-sided split on clean laws that differ in scale only.

h1v4.py, h1v4_s2.py and h1v4_s3.py are imported and not modified.
"""
from __future__ import annotations

import math

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T

N_GRID = S.N_GRID
R_X = S.R_X
FW_MIN_N = 1000
QS = (0.25, 0.5, 0.75, 0.9)

DICTS = {
    "oo3": (("o", 1.0), ("o", 3.0)),               # same parity, a_L = 1, a* = 3
    "oe3": (("o", 1.0), ("e", 3.0)),               # opposite parity
    "poly3": (("o", 1.0), ("e", 2.0), ("o", 3.0)),  # |I| = 2 when alpha_c < 4: reported only
}
SCALE_CLEAN = ((0.0, 1.3), (0.0, 1.0))             # S4-E: the classes differ in scale only


def cleans_of(src: str):
    """(train, test) clean laws, each a (plus, minus) pair."""
    if src == "gauss":
        g = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))
        return g, g
    if src == "scale":
        g = (S.Gauss(*SCALE_CLEAN[0]), S.Gauss(*SCALE_CLEAN[1]))
        return g, g
    if src == "rfly":
        P = S.rfly_pools(H.seed_of({"rfly_split": 1}))
        return P["train"], P["test"]
    raise ValueError(src)


def n_inadmissible(cols, alpha: float) -> int:
    return sum(2 * a > alpha for _, a in cols)


# --------------------------------------------------------------------------- replications

def crossover_rep(args) -> dict:
    """One training replication for a new S4 cell (blocks A, B, C, E). Same steps as
    h1v4_s3.crossover_rep, with the clean laws chosen by `src` and train/test pools kept apart."""
    (cols, astar, alpha, n, eps_tr, eps_te, src, seed) = args
    cols = [tuple(c) for c in cols]
    cont = T.Pareto(alpha)
    tr, te = cleans_of(src)
    rng = np.random.default_rng(seed)
    xp = S.draw(rng, n, eps_tr[0], tr[0], cont)
    xm = S.draw(rng, n, eps_tr[1], tr[1], cont)
    K, KW, KS, thr, cond = S.fit_all(cols, xp, xm)
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(KW)) and np.all(np.isfinite(KS))):
        return {"failed": 1.0}
    a = S.window_auc(rng, cols, K, KW, KS, thr, eps_te, te, cont)
    out = {"failed": 0.0, "Kt": float(K[-1]), "K": K.tolist(), "cond": cond,
           "N_on": H.onset(N_GRID, a["K"], 0.75), "peak": float(np.max(a["K"])),
           "auc_end": float(a["K"][-1]), "aucW_end": float(a["KW"][-1]),
           "aucS_end": float(a["KS"][-1]),
           "curve_K": a["K"].tolist(), "curve_KW": a["KW"].tolist(), "curve_KS": a["KS"].tolist()}
    if n_inadmissible(cols, alpha) == 1:
        ks = S.ksub(cols, eps_tr, tr, cont)
        out["lock"] = float(np.sign(K[-1]) == -np.sign(ks[-1]))
        out["Nstar"] = T.nstar(cols, K, eps_te, te, cont)
        w10, s10 = S.clean_auc10(rng, cols, KW, KS, thr, te)
        out["aucW10_clean"], out["aucS10_clean"] = w10, s10
    return out


def _pool(fn, jobs, workers=8):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs)


def _seeds(seed: int) -> list[int]:
    return [int(c.generate_state(1)[0]) for c in np.random.SeedSequence(seed).spawn(R_X)]


# --------------------------------------------------------------------------- summaries

def _q(x) -> list[float]:
    x = np.asarray(x, float)
    return [float(v) for v in np.quantile(x, QS)] if x.size else [float("nan")] * len(QS)


def summarize(reps, cols, astar, alpha, eps_tr, eps_te, cont, tr, te) -> dict:
    """The run-3 cell summary (same keys, same formulas) plus the per-fit values, their quartiles
    and the fixed-window check."""
    ok = [r for r in reps if r["failed"] == 0.0]
    gam = alpha / astar
    lam_pow = H.lambda_mid(gam, 0.75) ** (alpha / (astar - alpha))
    beta = S.beta_te(cols, eps_te, cont)
    Z = S.lepage(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Z": gam, "beta": beta})))
    on = np.array([r["N_on"] for r in ok])
    ns = np.array([r["Nstar"] for r in ok])
    sg = np.array([math.copysign(1.0, r["Kt"]) for r in ok])
    fin = np.isfinite(on)
    pred = np.array([T.predict(Z, N_GRID[-1] / s, g, astar, alpha) for s, g in zip(ns, sg)])
    end = np.array([r["auc_end"] for r in ok])
    err = np.abs(end - pred)
    ratio = on[fin] / (lam_pow * ns[fin])
    ksub = S.ksub(cols, eps_tr, tr, cont)
    mu_te = float(ksub @ S.test_mean_diff_L(cols, eps_te, te, cont))
    med = lambda key: float(np.median([r[key] for r in ok]))
    curves = np.array([r["curve_K"] for r in ok])
    fw_N = [int(N) for N in N_GRID if N >= FW_MIN_N]
    fw_auc, fw_psi = [], []
    for N in fw_N:
        j = int(np.flatnonzero(N_GRID == N)[0])
        fw_auc.append(float(curves[:, j].mean()))
        fw_psi.append(float(np.mean([T.predict(Z, N / s, g, astar, alpha) for s, g in zip(ns, sg)])))
    out = {"failed": float(len(reps) - len(ok)), "beta_te": beta, "lambda_m_pow": lam_pow,
           "ksub_lstar": float(ksub[-1]), "mu_te_L": mu_te,
           "medPeak": med("peak"), "onFinite": float(fin.mean()),
           "medNon": float(np.median(on[fin])) if fin.any() else float("nan"),
           "medNstar": float(np.median(ns)),
           "ratio": float(np.median(ratio)) if fin.any() else float("nan"),
           "medAucEnd": float(np.median(end)), "predErr": float(np.median(err)),
           "medPred": float(np.median(pred)),
           "medAucWEnd": med("aucW_end"), "medAucSEnd": med("aucS_end"),
           "lock": float(np.mean([r["lock"] for r in ok])),
           "pos": float(np.mean([r["Kt"] > 0 for r in ok])), "medCond": med("cond"),
           # dispersion over fits (DESIGN_S4.md, block D)
           "ratioQ": _q(ratio), "errQ": _q(err), "NstarQ": _q(ns), "endQ": _q(end),
           "errMax": float(err.max()) if err.size else float("nan"),
           "fit_Kt": [r["Kt"] for r in ok], "fit_Nstar": ns.tolist(), "fit_Non": on.tolist(),
           "fit_end": end.tolist(), "fit_pred": pred.tolist(),
           # fixed-window check of Cor. detwindow
           "fw_N": fw_N, "fw_meanAuc": fw_auc, "fw_meanPsi": fw_psi,
           "fwMaxDiff": float(np.max(np.abs(np.subtract(fw_auc, fw_psi)))),
           "N_grid": N_GRID.tolist(),
           "medCurveK": np.median(curves, axis=0).tolist(),
           "medCurveKW": np.median([r["curve_KW"] for r in ok], axis=0).tolist(),
           "medCurveKS": np.median([r["curve_KS"] for r in ok], axis=0).tolist()}
    if all("aucW10_clean" in r for r in ok):
        out["medCostW10"] = float(np.median([r["aucW10_clean"] - r["aucS10_clean"] for r in ok]))
    return out


def summarize_explore(reps, cols) -> dict:
    """|I| = 2 cells: no N*, no K^sub; the AUC path, the onset and the signs of the two top
    coefficients."""
    ok = [r for r in reps if r["failed"] == 0.0]
    on = np.array([r["N_on"] for r in ok])
    fin = np.isfinite(on)
    Ks = np.array([r["K"] for r in ok])
    return {"failed": float(len(reps) - len(ok)), "medPeak": float(np.median([r["peak"] for r in ok])),
            "onFinite": float(fin.mean()),
            "medNon": float(np.median(on[fin])) if fin.any() else float("nan"),
            "NonQ": _q(on[fin]), "medAucEnd": float(np.median([r["auc_end"] for r in ok])),
            "endQ": _q([r["auc_end"] for r in ok]),
            "medAucWEnd": float(np.median([r["aucW_end"] for r in ok])),
            "pos_K2": float(np.mean(Ks[:, -2] > 0)), "pos_K3": float(np.mean(Ks[:, -1] > 0)),
            "medCond": float(np.median([r["cond"] for r in ok])),
            "N_grid": N_GRID.tolist(),
            "medCurveK": np.median([r["curve_K"] for r in ok], axis=0).tolist(),
            "medCurveKW": np.median([r["curve_KW"] for r in ok], axis=0).tolist()}


# --------------------------------------------------------------------------- cells

def new_cell(dict_name, astar, alpha, n, eps_tr, eps_te, src, seed) -> dict:
    cols = DICTS[dict_name] if dict_name in DICTS else tuple(T.columns(dict_name, astar, alpha))
    jobs = [(cols, astar, alpha, n, eps_tr, eps_te, src, s) for s in _seeds(seed)]
    reps = _pool(crossover_rep, jobs)
    if n_inadmissible(cols, alpha) != 1:
        return summarize_explore(reps, cols)
    tr, te = cleans_of(src)
    out = summarize(reps, [tuple(c) for c in cols], astar, alpha, eps_tr, eps_te, T.Pareto(alpha), tr, te)
    if src == "rfly":
        P = S.rfly_pools(H.seed_of({"rfly_split": 1}))
        out.update({"n_flights": float(P["n_flights"]), "cache_sha256": P["cache_sha256"]})
    return out


def rerun_s3(orig: dict, eps_tr, eps_te) -> dict:
    """A run-3 crossover cell, same seeds, same replication function (h1v4_s3.crossover_rep)."""
    name, astar, alpha, n = orig["dict"], orig["astar"], orig["alpha"], orig["n"]
    jobs = [(name, astar, alpha, n, eps_tr, eps_te, s) for s in _seeds(H.seed_of(orig))]
    reps = _pool(T.crossover_rep, jobs)
    g = T.G_CLEAN
    return summarize(reps, T.columns(name, astar, alpha), astar, alpha, eps_tr, eps_te,
                     T.Pareto(alpha), g, g)


def rerun_s2(orig: dict, eps_te) -> dict:
    """A run-2 crossover cell, same seeds, same replication function (h1v4_s2.crossover_rep)."""
    name, astar, n, cont_kind, src = orig["dict"], orig["astar"], orig["n"], orig["cont"], orig["src"]
    jobs = [(name, astar, n, eps_te, cont_kind, src, s) for s in _seeds(H.seed_of(orig))]
    reps = _pool(S.crossover_rep, jobs)
    if src == "rfly":
        P = S.rfly_pools(H.seed_of({"rfly_split": 1}))
        tr, te = P["train"], P["test"]
    else:
        tr = te = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))
    cont = S.Contaminant(cont_kind)
    return summarize(reps, H.columns(name, astar), astar, S.ALPHA, (0.05, 0.05), eps_te, cont, tr, te)
