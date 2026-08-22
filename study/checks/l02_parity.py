#!/usr/bin/env python3
"""LC2: parity is necessary for discrimination (L2) + diagnostic curves d_nu(q), d_sigma(q)."""
from __future__ import annotations

import numpy as np

from _common import SEED, write_result
from mcp import basis as B, eval as E, panel
from mcp import projection as P
from mcp.moments import OneSidedMoments

rng = np.random.default_rng(SEED)
A = [0.6, 1.2, 1.4]   # never 1.0: sgn(x)|x|^1 duplicates the linear column
ALPHAS = np.linspace(0.05, 0.99, 40); ALPHAS = ALPHAS[np.abs(ALPHAS - 0.5) > 0.04]
N, NW = 256, 2000
QGRID = np.linspace(0.2, 3.0, 29)
out = {"A": A, "N": N, "NW": NW, "alphas": ALPHAS.tolist(), "parity": {}, "patp_auc": {}, "curves": {},
       "resampling_unit": "window (i.i.d. MC replicate)"}

sym_pairs = {**panel.lp1(), **panel.lp2()}
for name, (lm, ln) in sym_pairs.items():
    om, on = OneSidedMoments(lm), OneSidedMoments(ln)
    cols = B.mixed(A)
    out["parity"][name] = P.parity_split(cols, P.population_moments(cols, om), P.population_moments(cols, on))
    # feasible PATP (odd-only) AUC over alpha: must stay at chance
    tr1, tr2 = lm.rvs(NW * N, rng).reshape(NW, N), ln.rvs(NW * N, rng).reshape(NW, N)
    te1, te2 = lm.rvs(NW * N, rng).reshape(NW, N), ln.rvs(NW * N, rng).reshape(NW, N)
    mu, sd = E.standardizer(tr1, tr2)
    aucs = []
    for a in ALPHAS:
        pc = B.patp(a, 3)
        d = P.deflection(P.sample_moments(pc, (tr1 - mu) / sd), P.sample_moments(pc, (tr2 - mu) / sd))
        aucs.append(E.auc(E.window_scores(pc, d.K, te1, mu, sd), E.window_scores(pc, d.K, te2, mu, sd)))
    se = float(np.sqrt((2 * NW + 1) / (12.0 * NW * NW)))   # SE of AUC at chance (Hanley-McNeil, n1=n2=NW)
    out["patp_auc"][name] = {"auc": aucs, "max_auc": max(aucs), "se": se, "at_chance": max(aucs) <= 0.5 + 2 * se}

all_pairs = {**sym_pairs, **panel.lp3(), **panel.lp4(), "LP5 drone vs pult (eps=0)": panel.lp5(0.0)}
for name, (lm, ln) in all_pairs.items():
    om, on = OneSidedMoments(lm), OneSidedMoments(ln)
    out["curves"][name] = {"q": QGRID.tolist(),
                           "d_nu": [on.nu_total(q) - om.nu_total(q) for q in QGRID],
                           "d_sigma": [on.sigma(q) - om.sigma(q) for q in QGRID]}

out["PASS"] = all(abs(v["J_full"] - v["J_even"]) < 1e-10 and v["J_odd"] < 1e-10 for v in out["parity"].values()) \
    and all(v["at_chance"] for v in out["patp_auc"].values())
print({k: (v["J_full"], v["J_even"], v["J_odd"]) for k, v in out["parity"].items()})
print({k: (v["max_auc"], v["at_chance"]) for k, v in out["patp_auc"].items()}, "PASS", out["PASS"])
print("written", write_result("l02", out))
