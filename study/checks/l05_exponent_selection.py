#!/usr/bin/env python3
"""LC5: exponent selection by J beyond the PATP path (L4, L6): oracle and feasible (validation split)."""
from __future__ import annotations

import numpy as np

from _common import SEED, write_result
from mcp import basis as B, eval as E, panel
from mcp import projection as P
from mcp.moments import OneSidedMoments

ALPHAS = np.linspace(0.05, 0.99, 40); ALPHAS = ALPHAS[np.abs(ALPHAS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 2.51, 0.1), 2)
PAIRS_FREE = [(a, b) for i, a in enumerate(AGRID) for b in AGRID[i + 1:]]
N, NW, NTR, NVAL, R = 256, 4000, 1_000_000, 500_000, 30
FEAS_AGRID = np.round(np.arange(0.3, 2.51, 0.3), 2)
FEAS_PAIRS = [(a, b) for i, a in enumerate(FEAS_AGRID) for b in FEAS_AGRID[i + 1:]]
DICTS = {"P": (lambda al: B.patp(al, 3), [(al,) for al in ALPHAS], [(al,) for al in ALPHAS]),
         "free_signed": (lambda a, b: B.free_signed([a, b]), PAIRS_FREE, FEAS_PAIRS),
         "mixed": (lambda a: B.mixed([a]), [(a,) for a in AGRID], [(a,) for a in FEAS_AGRID])}


def argmax_J(cm_fn, cn_fn, builder, grid):
    best = (-np.inf, None, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(cm_fn(cols), cn_fn(cols))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args, d.cond)
    return best


pairs = {**panel.lp3(), "LP5 drone vs pult (eps=0)": panel.lp5(0.0)}
out = {"N": N, "NW": NW, "NTR": NTR, "NVAL": NVAL, "R": R, "oracle": {}, "feasible": {},
       "resampling_unit": "seed (independent MC replicate); within-seed AUC over NW windows"}
rng = np.random.default_rng(SEED)
for name, (lm, ln) in pairs.items():
    om, on = OneSidedMoments(lm), OneSidedMoments(ln)
    orc = {}
    for k, (builder, grid, _) in DICTS.items():
        J, args, cond = argmax_J(lambda c: P.population_moments(c, om), lambda c: P.population_moments(c, on), builder, grid)
        orc[k] = {"J": J, "args": [float(a) for a in args], "cond": cond}
    for k in ("free_signed", "mixed"):
        orc[f"gain_{k}_pct"] = 100 * (orc[k]["J"] - orc["P"]["J"]) / orc["P"]["J"]
    out["oracle"][name] = orc
    print(name, {k: (round(v["J"], 5), v["args"]) if isinstance(v, dict) else round(v, 2) for k, v in orc.items()})

    feas = {k: {"auc_by_seed": [], "selected_args_by_seed": []} for k in DICTS}
    for r in range(R):
        x1, x2 = lm.rvs(NTR, rng), ln.rvs(NTR, rng)
        v1, v2 = lm.rvs(NVAL, rng), ln.rvs(NVAL, rng)
        te1, te2 = lm.rvs(NW * N, rng).reshape(NW, N), ln.rvs(NW * N, rng).reshape(NW, N)
        mu, sd = E.standardizer(x1, x2)
        z1, z2, w1, w2 = (x1 - mu) / sd, (x2 - mu) / sd, (v1 - mu) / sd, (v2 - mu) / sd
        for k, (builder, _, fgrid) in DICTS.items():
            _, args, _ = argmax_J(lambda c: P.sample_moments(c, w1), lambda c: P.sample_moments(c, w2), builder, fgrid)
            cols = builder(*args)
            K = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2)).K
            feas[k]["auc_by_seed"].append(E.auc(E.window_scores(cols, K, te1, mu, sd), E.window_scores(cols, K, te2, mu, sd)))
            feas[k]["selected_args_by_seed"].append([float(a) for a in args])
    for k in DICTS:
        a = np.array(feas[k]["auc_by_seed"]); feas[k]["auc_mean"], feas[k]["auc_sd"] = float(a.mean()), float(a.std(ddof=1))
    for k in ("free_signed", "mixed"):
        d = np.array(feas[k]["auc_by_seed"]) - np.array(feas["P"]["auc_by_seed"])
        bs = [np.mean(rng.choice(d, size=R)) for _ in range(2000)]
        feas[f"diff_{k}_minus_P"] = {"mean": float(d.mean()), "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))}
    out["feasible"][name] = feas
    print("  feasible", {k: (round(feas[k]["auc_mean"], 4), round(feas[k]["auc_sd"], 4)) for k in DICTS})
out["PASS"] = True   # descriptive check: reported as measured
print("written", write_result("l05", out))
