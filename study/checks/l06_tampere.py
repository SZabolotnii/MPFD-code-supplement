#!/usr/bin/env python3
"""LC6: real pair Parrot mambo control (remote) vs video (drone), Tampere dataset. Illustration only:
n_recordings = 1 per class; two time splits; moving-block bootstrap CI over windows."""
from __future__ import annotations

import numpy as np

from _common import SEED, write_result
from mcp import basis as B, eval as E
from mcp import projection as P
from mcp.data import tampere as T
from mcp.moments import hill_tail_index

if not T.available():
    raise SystemExit("Tampere recordings not found under SDR_Dron/data/raw — LC6 skipped")
rng = np.random.default_rng(SEED)
N, GATE, MAXW, BLOCK, NBOOT = 1024, 3.0, 20000, 16, 200
ALPHAS = np.linspace(0.05, 0.99, 40); ALPHAS = ALPHAS[np.abs(ALPHAS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 1.91, 0.1), 2)
HGRID = np.round(np.linspace(0.55, 1.95, 8), 3)

halves = {"train first half / test second": ((0.0, 0.5), (0.5, 1.0)), "train second half / test first": ((0.5, 1.0), (0.0, 0.5))}
out = {"N": N, "gate": GATE, "max_windows": MAXW, "block": BLOCK, "NBOOT": NBOOT, "n_recordings_per_class": 1,
       "resampling_unit": "moving block of windows (time-ordered); no split-level test possible", "splits": {}}


def select(builder, grid, z1, z2):
    best = (-np.inf, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args)
    return best[1]


for sname, ((a0, a1), (b0, b1)) in halves.items():
    tr1, tr2 = T.gated_real_windows(T.CONTROL, N, GATE, a0, a1, MAXW), T.gated_real_windows(T.VIDEO, N, GATE, a0, a1, MAXW)
    te1, te2 = T.gated_real_windows(T.CONTROL, N, GATE, b0, b1, MAXW), T.gated_real_windows(T.VIDEO, N, GATE, b0, b1, MAXW)
    mu, sd = E.standardizer(tr1, tr2)
    z1, z2 = (tr1 - mu) / sd, (tr2 - mu) / sd
    alpha_hat = hill_tail_index(np.concatenate([z1.ravel(), z2.ravel()]), 0.02)
    astar = select(lambda al: B.patp(al, 3), [(al,) for al in ALPHAS], z1, z2)
    mstar = select(lambda a: B.mixed([a]), [(a,) for a in AGRID], z1, z2)
    hstar = select(lambda a, b: B.halfline([a], [b]), [(a, b) for a in HGRID for b in HGRID], z1, z2)
    dicts = {"linear {x}": B.classical(1), "classical {x,x2,x3}": B.classical(3), f"PATP a*={astar[0]:.2f}": B.patp(astar[0], 3),
             f"mixed a*={mstar[0]:.2f}": B.mixed([mstar[0]]), f"halfline {hstar}": B.halfline([hstar[0]], [hstar[1]])}
    rows = {}
    for k, cols in dicts.items():
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        s1, s2 = E.window_scores(cols, d.K, te1, mu, sd), E.window_scores(cols, d.K, te2, mu, sd)
        lo, hi = E.moving_block_bootstrap(s1, s2, BLOCK, NBOOT, rng)
        rows[k] = {"auc": E.auc(s1, s2), "ci_lo": lo, "ci_hi": hi, "J_train": d.J, "cond": d.cond}
    out["splits"][sname] = {"n_train": [int(tr1.shape[0]), int(tr2.shape[0])], "n_test": [int(te1.shape[0]), int(te2.shape[0])],
                            "hill_alpha_hat": alpha_hat, "ceiling": alpha_hat / 2, "rows": rows}
    print(sname, out["splits"][sname]["n_train"], out["splits"][sname]["n_test"], "alpha_hat", round(alpha_hat, 2))
    for k, r in rows.items():
        print(f"   {k:<32} AUC={r['auc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  cond={r['cond']:.1e}")
out["PASS"] = True   # descriptive
print("written", write_result("l06", out))
