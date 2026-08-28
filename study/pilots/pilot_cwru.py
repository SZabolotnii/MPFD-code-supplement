#!/usr/bin/env python3
"""Pilot P2: CWRU healthy vs seeded-fault vibration, drive-end 12 kHz, same protocol and criteria
as PILOTS_2026-08-28.md. Recording = one (fault, load) file; leave-one-load-out over the
four motor loads, so the held-out unit is a recording. Writes pilot_cwru.json beside itself."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # study/ -- these live in study/pilots/
from mcp import basis as B, eval as E, projection as P  # noqa: E402
from mcp.moments import hill_tail_index  # noqa: E402

DATA = Path(os.environ.get("RESEARCH_DATA_ROOT", Path.home() / "Project/datasets")) / "store/cwru-bearing/data"
SEED, NBOOT, N = 20260821, 200, 1024
TAUS = np.linspace(0.05, 0.99, 40); TAUS = TAUS[np.abs(TAUS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 1.91, 0.1), 2)
HGRID = np.round(np.linspace(0.55, 1.95, 8), 3)
HEALTHY = [97, 98, 99, 100]                       # loads 0..3 hp
FAULTS = {"IR007": [105, 106, 107, 108], "B007": [118, 119, 120, 121], "OR007": [130, 131, 132, 133]}


def windows(fid: int) -> np.ndarray:
    d = loadmat(DATA / f"{fid}.mat")
    x = np.asarray(d[f"X{fid:03d}_DE_time"], float).ravel()
    return x[: len(x) // N * N].reshape(-1, N)


def select(builder, grid, z1, z2):
    best = (-np.inf, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args)
    return best[1]


rng = np.random.default_rng(SEED)
out = {"N": N, "resampling_unit": "recording (leave-one-load-out, 4 folds); window bootstrap within fold", "pairs": {}}
for fname, fids in FAULTS.items():
    folds = []
    for held in range(4):
        tr1 = np.concatenate([windows(HEALTHY[i]) for i in range(4) if i != held])
        tr2 = np.concatenate([windows(fids[i]) for i in range(4) if i != held])
        te1, te2 = windows(HEALTHY[held]), windows(fids[held])
        mu, sd = E.standardizer(tr1, tr2)
        z1, z2 = (tr1 - mu) / sd, (tr2 - mu) / sd
        tstar = select(lambda t: B.patp(t, 3), [(t,) for t in TAUS], z1, z2)
        astar = select(lambda a: B.mixed([a]), [(a,) for a in AGRID], z1, z2)
        hstar = select(lambda a, b: B.halfline([a], [b]), [(a, b) for a in HGRID for b in HGRID], z1, z2)
        estar = select(lambda a, b: [B.LINEAR, B.Column("abs", float(a)), B.Column("abs", float(b))],
                       [(a, b) for i, a in enumerate(HGRID) for b in HGRID[i + 1:]], z1, z2)
        dicts = {"linear": B.classical(1), "classical": B.classical(3), "P": B.patp(tstar[0], 3),
                 "M": B.mixed([astar[0]]), "H": B.halfline([hstar[0]], [hstar[1]]),
                 "E": [B.LINEAR, B.Column("abs", float(estar[0])), B.Column("abs", float(estar[1]))]}
        fold = {"held_load": held, "n_test": [int(te1.shape[0]), int(te2.shape[0])],
                "hill_alpha_pooled": hill_tail_index(np.concatenate([z1.ravel(), z2.ravel()]), 0.02),
                "kurt": {k: float(((v - v.mean()) / v.std()).ravel().__pow__(4).mean()) for k, v in (("healthy", z1), ("fault", z2))},
                "skew": {k: float(((v - v.mean()) / v.std()).ravel().__pow__(3).mean()) for k, v in (("healthy", z1), ("fault", z2))},
                "args": {"P": float(tstar[0]), "M": float(astar[0]), "H": [float(hstar[0]), float(hstar[1])], "E": [float(estar[0]), float(estar[1])]},
                "rows": {}}
        scores = {}
        for k, cols in dicts.items():
            cm, cn = P.sample_moments(cols, z1), P.sample_moments(cols, z2)
            d = P.deflection(cm, cn)
            s1, s2 = E.window_scores(cols, d.K, te1, mu, sd), E.window_scores(cols, d.K, te2, mu, sd)
            scores[k] = (s1, s2)
            fold["rows"][k] = {"auc": E.auc(s1, s2), "J_train": d.J, "cond": d.cond, "parity": P.parity_split(cols, cm, cn)}
        best = max(("P", "M", "H", "E"), key=lambda k: fold["rows"][k]["auc"])
        s1b, s2b = scores[best]; s1c, s2c = scores["classical"]
        n = min(len(s1b), len(s2b)); idx = rng.integers(0, n, size=(NBOOT, n))
        diffs = np.array([E.auc(s1b[i], s2b[i]) - E.auc(s1c[i], s2c[i]) for i in idx])
        fold["best"] = best
        fold["diff"] = {"point": fold["rows"][best]["auc"] - fold["rows"]["classical"]["auc"],
                        "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}
        folds.append(fold)
        r = fold["rows"]
        print(f"{fname} held load {held}: alpha {fold['hill_alpha_pooled']:.2f} kurt h/f {fold['kurt']['healthy']:.2f}/{fold['kurt']['fault']:.2f} | "
              + " ".join(f"{k}={r[k]['auc']:.3f}" for k in r) + f" | best {best} diff {fold['diff']['point']:+.3f} [{fold['diff']['ci_lo']:+.3f},{fold['diff']['ci_hi']:+.3f}]")
    pm = [f["rows"]["M"]["parity"] for f in folds]
    even_share = float(np.mean([p["J_even"] / p["J_full"] for p in pm]))
    crit = {"1_mechanism": bool(np.mean([f["hill_alpha_pooled"] for f in folds]) < 4 or even_share > 0.5),
            "2_gain_over_control_all_folds": bool(all(f["diff"]["ci_lo"] > 0 for f in folds)),
            "3_nontrivial": bool(np.mean([f["rows"][f["best"]]["auc"] for f in folds]) >= 0.60)}
    out["pairs"][fname] = {"folds": folds, "even_share_M": even_share, "criteria": crit, "PASS": all(crit.values())}
    print(f"  => {fname}: even share {even_share:.2f}  {crit}  PASS {all(crit.values())}")
(HERE / "pilot_cwru.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
