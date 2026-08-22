#!/usr/bin/env python3
"""LC3: admissibility as the robustness mechanism (L3). Migrates verification/patp_parity_control.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import SEED, STUDY, write_result
from mcp import basis as B, eval as E, panel
from mcp import projection as P
from mcp.moments import InadmissibleMoment, OneSidedMoments, hill_tail_index

rng = np.random.default_rng(SEED)
N, NW, NBOOT = 256, 4000, 200
EPS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
NS = [1_000, 10_000, 100_000, 1_000_000]
R = 30
DICTS = {"linear {x}": B.classical(1), "classical {x,x2}": B.classical(2), "classical {x,x2,x3}": B.classical(3),
         "PATP a=0.99": B.patp(0.99, 2), "PATP a=0.31": B.patp(0.31, 2), "mixed {x,|x|^.75,sgn|x|^.75}": B.mixed([0.75])}

# (a) admissibility at alpha_min = 3 (t(3) contamination; at t(2) even the linear column sits on the boundary), programmatic
DF_POP = 3
lm, ln = panel.lp5(0.10, df=DF_POP)
om, on = OneSidedMoments(lm), OneSidedMoments(ln)
adm = {}
for name, cols in DICTS.items():
    try:
        P.population_moments(cols, om); adm[name] = True
    except InadmissibleMoment:
        adm[name] = False
assert adm == {k: B.admissible(c, float(DF_POP)) for k, c in DICTS.items()}

# (b) empirical J vs n at t(3): classical(3) does not converge, mixed does (monotone; no rate claim — x has no root-n at alpha=3)
jvn = {k: {"n": NS, "median": [], "iqr": []} for k in ("classical {x,x2,x3}", "mixed {x,|x|^.75,sgn|x|^.75}")}
for n in NS:
    js = {k: [] for k in jvn}
    for _ in range(R):
        x1, x2 = lm.rvs(n, rng), ln.rvs(n, rng)
        mu, sd = E.standardizer(x1, x2)
        for k in jvn:
            js[k].append(P.deflection(P.sample_moments(DICTS[k], (x1 - mu) / sd), P.sample_moments(DICTS[k], (x2 - mu) / sd)).J)
    for k in jvn:
        q25, q50, q75 = np.percentile(js[k], [25, 50, 75])
        jvn[k]["median"].append(float(q50)); jvn[k]["iqr"].append(float(q75 - q25))
for k in jvn:
    jvn[k]["ratio"] = [i / m for i, m in zip(jvn[k]["iqr"], jvn[k]["median"])]
cl, mx = jvn["classical {x,x2,x3}"]["ratio"], jvn["mixed {x,|x|^.75,sgn|x|^.75}"]["ratio"]
conv = {"classical_nonconvergent": cl[3] >= cl[1], "mixed_monotone": all(mx[i + 1] < mx[i] for i in range(3))}

# (c) AUC table by eps at t(2) (weights from clean training, as in the frozen script; equivalence with the frozen JSON)
d0, p0 = panel.lp5(0.0)
tr1, tr2 = d0.rvs(NW * N, rng).reshape(NW, N), p0.rvs(NW * N, rng).reshape(NW, N)
mu, sd = E.standardizer(tr1, tr2)
K = {k: P.deflection(P.sample_moments(c, (tr1 - mu) / sd), P.sample_moments(c, (tr2 - mu) / sd)).K for k, c in DICTS.items()}
table = {k: [] for k in DICTS}
for eps in EPS:
    dm, pm = panel.lp5(eps)
    te1, te2 = dm.rvs(NW * N, rng).reshape(NW, N), pm.rvs(NW * N, rng).reshape(NW, N)
    for k, c in DICTS.items():
        s1, s2 = E.window_scores(c, K[k], te1, mu, sd), E.window_scores(c, K[k], te2, mu, sd)
        lo, hi = E.paired_window_bootstrap(s1, s2, NBOOT, rng)
        table[k].append({"auc": E.auc(s1, s2), "ci_lo": lo, "ci_hi": hi})

# (d) Hill ceiling vs J-optimal exponent of mixed({a}) at eps = 0.10; cond-gated per global constraints
# (mixed([1.0]) contains sgn(x)|x|^1 == x, a duplicate of the linear column -> F singular -> J is nan/garbage there)
dm, pm = panel.lp5(0.10)
xt = np.concatenate([dm.rvs(NW * N, rng), pm.rvs(NW * N, rng)])
alpha_hat = hill_tail_index(xt, frac=0.05)
grid = np.round(np.arange(0.3, 1.91, 0.1), 2)
x1, x2 = dm.rvs(NW * N, rng), pm.rvs(NW * N, rng)
mu2, sd2 = E.standardizer(x1, x2)
defl_g = [P.deflection(P.sample_moments(B.mixed([a]), (x1 - mu2) / sd2), P.sample_moments(B.mixed([a]), (x2 - mu2) / sd2)) for a in grid]
cond_g = [d.cond for d in defl_g]
ok_g = [bool(np.isfinite(d.J) and c <= P.COND_LIMIT) for d, c in zip(defl_g, cond_g)]
Jg = [d.J if ok else float("nan") for d, ok in zip(defl_g, ok_g)]
excluded_a = [float(a) for a, ok in zip(grid, ok_g) if not ok]
valid_idx = [i for i, ok in enumerate(ok_g) if ok]
argmax_a = float(grid[valid_idx[int(np.argmax([Jg[i] for i in valid_idx]))]])

# (e) equivalence with the frozen verification JSON (same design, different RNG stream)
frozen = json.loads((STUDY.parent / "verification" / "results_patp_parity_control.json").read_text())
fmap = {"linear {x}": "лінійний {x}", "classical {x,x2}": "класичний {x, x²}", "classical {x,x2,x3}": "класичний {x, x², x³}",
        "PATP a=0.99": "PATP α=0.99 {x, sign|x|²}", "PATP a=0.31": "PATP α*=0.31 {x, sign|x|^0.75}",
        "mixed {x,|x|^.75,sgn|x|^.75}": "змішана парність {x, |x|^0.75, sign|x|^0.75}"}
dev = 0.0
for k, fk in fmap.items():
    for i, eps in enumerate(frozen["eps"]):
        dev = max(dev, abs(table[k][EPS.index(eps)]["auc"] - frozen["auc"][fk][i]["auc"]))

collapse = all(0.45 <= table[k][i]["auc"] <= 0.55 for k in ("classical {x,x2}", "classical {x,x2,x3}") for i, e in enumerate(EPS) if e >= 0.05)

# diagnostics (additions only; no threshold, seed, size or scenario changed)
cm3, cn3 = P.population_moments(B.mixed([0.75]), om), P.population_moments(B.mixed([0.75]), on)
J_pop_mixed_df3 = P.deflection(cm3, cn3).J
rel_err_mixed = [abs(m - J_pop_mixed_df3) / J_pop_mixed_df3 for m in jvn["mixed {x,|x|^.75,sgn|x|^.75}"]["median"]]
classical_median_collapse = jvn["classical {x,x2,x3}"]["median"][1] / jvn["classical {x,x2,x3}"]["median"][3]
mixed_median_drop = jvn["mixed {x,|x|^.75,sgn|x|^.75}"]["median"][1] / jvn["mixed {x,|x|^.75,sgn|x|^.75}"]["median"][3]
equiv_cells = {}
n_within, n_cells = 0, 0
for k, fk in fmap.items():
    equiv_cells[k] = []
    for i, eps in enumerate(frozen["eps"]):
        cell = table[k][EPS.index(eps)]
        frozen_auc = frozen["auc"][fk][i]["auc"]
        within = bool(cell["ci_lo"] <= frozen_auc <= cell["ci_hi"])
        equiv_cells[k].append(within)
        n_cells += 1
        n_within += int(within)
diagnostics = {"J_pop_mixed_df3": J_pop_mixed_df3, "rel_err_mixed": rel_err_mixed,
               "classical_median_collapse": classical_median_collapse, "mixed_median_drop": mixed_median_drop,
               "equivalence_within_ci": {"cells": equiv_cells, "n_within": n_within, "n_cells": n_cells},
               "reading": "registered criteria unchanged; see CHECKS.md post-run annotation"}

out = {"N": N, "NW": NW, "NBOOT": NBOOT, "R": R, "eps": EPS, "resampling_unit": "window (i.i.d. MC replicate)",
       "df_population": DF_POP, "df_auc_table": 2, "admissible": adm, "j_vs_n": jvn, "convergence": conv, "auc_table": table,
       "hill": {"alpha_hat": alpha_hat, "ceiling": alpha_hat / 2, "argmax_a": argmax_a, "grid": grid.tolist(), "J": Jg,
                "cond": cond_g, "excluded_a": excluded_a, "below_ceiling": argmax_a < alpha_hat / 2},
       "equivalence": {"max_abs_dAUC": dev, "pass": dev < 0.01}, "classical_collapse": collapse, "diagnostics": diagnostics}
out["PASS"] = conv["classical_nonconvergent"] and conv["mixed_monotone"] and collapse and out["equivalence"]["pass"] and out["hill"]["below_ceiling"]
for k in DICTS:
    print(f"{k:<34}" + "".join(f"  {r['auc']:.3f}[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]" for r in table[k]))
print("hill", alpha_hat, "ceiling", alpha_hat / 2, "argmax", argmax_a, "excluded_a", excluded_a, "| equivalence", dev, "| PASS", out["PASS"])
print("diagnostics: J_pop_mixed_df3", J_pop_mixed_df3, "rel_err_mixed", rel_err_mixed)
print("diagnostics: classical_median_collapse", classical_median_collapse, "mixed_median_drop", mixed_median_drop)
print("diagnostics: equivalence_within_ci n_within/n_cells", n_within, "/", n_cells)
print("written", write_result("l03", out))
