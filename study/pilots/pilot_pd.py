#!/usr/bin/env python3
"""Pilot P1: PD vs NonPD antenna segments (figshare 24033225). Protocol and pass criteria are
fixed in PILOTS_2026-08-28.md before this file was written. Writes pilot_pd.json beside itself
and nothing else; study/results/ is untouched."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # study/ -- these live in study/pilots/
from mcp import basis as B, eval as E, projection as P  # noqa: E402
from mcp.moments import hill_tail_index  # noqa: E402

DATA = Path(os.environ.get("RESEARCH_DATA_ROOT", Path.home() / "Project/datasets")) / "store/pd-antenna-signals/data"
SEED, NBOOT = 20260821, 200
TAUS = np.linspace(0.05, 0.99, 40); TAUS = TAUS[np.abs(TAUS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 1.91, 0.1), 2)
HGRID = np.round(np.linspace(0.55, 1.95, 8), 3)


def load(split: str):
    d = loadmat(DATA / f"{split}.mat")
    s = d[split][0, 0]["signals"]
    x = np.stack([np.asarray(c, np.float64).ravel() for c in s[:, 0]])
    blob = d["__function_workspace__"].tobytes()
    # categorical codes live in the MCOS blob as a run of exactly len(x) bytes in {1,2};
    # categoryNames order in the blob is NonPD, PD -> code 1 = NonPD, 2 = PD.
    names = [m.group() for m in re.finditer(rb"NonPD|PD", blob)]
    assert names[:2] == [b"NonPD", b"PD"], names
    a = np.frombuffer(blob, np.uint8)
    n = x.shape[0]
    hits = [i for i in range(len(a) - n + 1) if a[i] in (1, 2) and np.all(np.isin(a[i:i + n], (1, 2)))
            and (i == 0 or a[i - 1] not in (1, 2)) and (i + n == len(a) or a[i + n] not in (1, 2))]
    assert len(hits) == 1, hits
    lab = a[hits[0]:hits[0] + n]
    return x[lab == 1], x[lab == 2]          # NonPD (class m), PD (class n)


def select(builder, grid, z1, z2):
    best = (-np.inf, None, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args, d.cond)
    return best


rng = np.random.default_rng(SEED)
tr1, tr2 = load("Tr0")
te1, te2 = load("Te0")
mu, sd = E.standardizer(tr1, tr2)
z1, z2 = (tr1 - mu) / sd, (tr2 - mu) / sd
out = {"n_train": [int(tr1.shape[0]), int(tr2.shape[0])], "n_test": [int(te1.shape[0]), int(te2.shape[0])],
       "segment_len": int(tr1.shape[1]), "resampling_unit": "segment (400 samples); no session metadata"}
for name, z in (("NonPD", z1), ("PD", z2)):
    v = z.ravel(); v = (v - v.mean()) / v.std()
    out[f"moments_{name}"] = {"skew": float((v ** 3).mean()), "kurt": float((v ** 4).mean()),
                              "hill_alpha": hill_tail_index(v, 0.02)}
out["hill_alpha_pooled"] = hill_tail_index(np.concatenate([z1.ravel(), z2.ravel()]), 0.02)

Jp, tstar, _ = select(lambda t: B.patp(t, 3), [(t,) for t in TAUS], z1, z2)
Jm, astar, _ = select(lambda a: B.mixed([a]), [(a,) for a in AGRID], z1, z2)
Jh, hstar, _ = select(lambda a, b: B.halfline([a], [b]), [(a, b) for a in HGRID for b in HGRID], z1, z2)
Je, estar, _ = select(lambda a, b: [B.LINEAR, B.Column("abs", float(a)), B.Column("abs", float(b))],
                      [(a, b) for i, a in enumerate(HGRID) for b in HGRID[i + 1:]], z1, z2)
dicts = {"linear {x}": B.classical(1), "classical {x,x2,x3}": B.classical(3),
         f"P tau*={tstar[0]:.2f}": B.patp(tstar[0], 3), f"M a*={astar[0]:.2f}": B.mixed([astar[0]]),
         f"H {tuple(map(float, hstar))}": B.halfline([hstar[0]], [hstar[1]]),
         f"E {tuple(map(float, estar))}": [B.LINEAR, B.Column("abs", float(estar[0])), B.Column("abs", float(estar[1]))]}

scores, rows = {}, {}
for k, cols in dicts.items():
    cm, cn = P.sample_moments(cols, z1), P.sample_moments(cols, z2)
    d = P.deflection(cm, cn)
    s1, s2 = E.window_scores(cols, d.K, te1, mu, sd), E.window_scores(cols, d.K, te2, mu, sd)
    scores[k] = (s1, s2)
    lo, hi = E.paired_window_bootstrap(s1, s2, NBOOT, rng)
    rows[k] = {"auc": E.auc(s1, s2), "ci_lo": lo, "ci_hi": hi, "J_train": d.J, "cond": d.cond,
               "parity": P.parity_split(cols, cm, cn)}
out["rows"] = rows

# paired bootstrap of the AUC difference best-vs-control, same segment resample for both
ctrl = "classical {x,x2,x3}"
cands = [k for k in rows if k not in ("linear {x}", ctrl)]
best = max(cands, key=lambda k: rows[k]["auc"])
s1b, s2b = scores[best]; s1c, s2c = scores[ctrl]
n = min(len(s1b), len(s2b))
idx = rng.integers(0, n, size=(NBOOT, n))
diffs = np.array([E.auc(s1b[i], s2b[i]) - E.auc(s1c[i], s2c[i]) for i in idx])
out["best"] = best
out["diff_best_minus_control"] = {"point": rows[best]["auc"] - rows[ctrl]["auc"],
                                  "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}
pm = rows[f"M a*={astar[0]:.2f}"]["parity"]
out["criteria"] = {
    "1_mechanism": bool(out["hill_alpha_pooled"] < 4 or pm["J_even"] / pm["J_full"] > 0.5),
    "2_gain_over_control": bool(out["diff_best_minus_control"]["ci_lo"] > 0),
    "3_nontrivial": bool(rows[best]["auc"] >= 0.60)}
out["PASS"] = all(out["criteria"].values())

print("n", out["n_train"], out["n_test"], "alpha_hat pooled %.2f" % out["hill_alpha_pooled"])
for nm in ("NonPD", "PD"):
    print("  %-6s skew %+.3f kurt %.2f hill %.2f" % (nm, *out[f"moments_{nm}"].values()))
for k, r in rows.items():
    p = r["parity"]
    print(f"   {k:<28} AUC={r['auc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  J={r['J_train']:.4f} "
          f"(even {p['J_even']:.4f} / odd {p['J_odd']:.4f})  cond={r['cond']:.1e}")
print("best:", best, "diff vs control", out["diff_best_minus_control"])
print("criteria:", out["criteria"], "PASS", out["PASS"])
(HERE / "pilot_pd.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
