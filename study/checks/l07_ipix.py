#!/usr/bin/env python3
"""LC7: IPIX Dartmouth 1993 sea clutter, clutter-only cells (H0) vs the primary target cell (H1),
leave-one-file-out over the 14 target files. Registered in CHECKS.md on 2026-08-28 before this
file was written; protocol and grids follow the tmp/ pilot of the same day.

The one dictionary compared with the classical control on each held-out file is chosen on the
training files by J alone -- the test file never enters the choice. The recording-level gate
(2r) is the bootstrap CI of the mean AUC difference over the 14 files; the all-folds window
criterion (2w) of the pilot is reported, not gated."""
from __future__ import annotations

import numpy as np

from _common import SEED, write_result
from mcp import basis as B, eval as E
from mcp import projection as P
from mcp.data import ipix as D
from mcp.moments import hill_tail_index

if not D.available():
    raise SystemExit(f"IPIX files not found under {D.RAW} -- LC7 skipped")
rng = np.random.default_rng(SEED)
N, NBOOT, NBOOT_FILES = 1024, 200, 2000
TAUS = np.linspace(0.05, 0.99, 40); TAUS = TAUS[np.abs(TAUS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 1.91, 0.1), 2)
HGRID = np.round(np.linspace(0.55, 1.95, 8), 3)
CANDIDATES = ("M", "H", "E", "P")


def select(builder, grid, z1, z2):
    best = (-np.inf, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args)
    return best[1]


def moments(v):
    z = (v - v.mean()) / v.std()
    return {"skew": float((z ** 3).mean()), "kurt": float((z ** 4).mean()), "sd": float(v.std())}


recs = []
for stem, prim, sec in D.FILES:
    h0, h1, clutter = D.h0_h1(stem, prim, sec, N)
    recs.append({"name": stem, "h0": h0, "h1": h1, "primary": prim, "clutter_cells": clutter,
                 "stats": {"clutter": moments(h0), "target": moments(h1)}})

out = {"N": N, "NBOOT": NBOOT, "NBOOT_FILES": NBOOT_FILES, "n_recordings": len(recs),
       "resampling_unit": "file (leave-one-file-out); paired window bootstrap within a fold; bootstrap of the mean difference over files",
       "files": [{k: v for k, v in r.items() if k not in ("h0", "h1")} for r in recs], "folds": []}
for held in range(len(recs)):
    tr1 = np.concatenate([r["h0"] for i, r in enumerate(recs) if i != held])
    tr2 = np.concatenate([r["h1"] for i, r in enumerate(recs) if i != held])
    te1, te2 = recs[held]["h0"], recs[held]["h1"]
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
    fold = {"held": recs[held]["name"], "n_test": [int(te1.shape[0]), int(te2.shape[0])],
            "hill_alpha_pooled": hill_tail_index(np.concatenate([z1.ravel(), z2.ravel()]), 0.02),
            "args": {"P": float(tstar[0]), "M": float(astar[0]), "H": [float(hstar[0]), float(hstar[1])],
                     "E": [float(estar[0]), float(estar[1])]}, "rows": {}}
    scores = {}
    for k, cols in dicts.items():
        cm, cn = P.sample_moments(cols, z1), P.sample_moments(cols, z2)
        d = P.deflection(cm, cn)
        s1, s2 = E.window_scores(cols, d.K, te1, mu, sd), E.window_scores(cols, d.K, te2, mu, sd)
        scores[k] = (s1, s2)
        fold["rows"][k] = {"auc": E.auc(s1, s2), "J_train": d.J, "cond": d.cond, "parity": P.parity_split(cols, cm, cn)}
    # chosen on the training files only: the candidate with the largest training deflection
    chosen = max(CANDIDATES, key=lambda k: fold["rows"][k]["J_train"])
    s1b, s2b = scores[chosen]; s1c, s2c = scores["classical"]
    n = min(len(s1b), len(s2b))
    idx = rng.integers(0, n, size=(NBOOT, n))
    diffs = np.array([E.auc(s1b[i % len(s1b)], s2b[i % len(s2b)]) - E.auc(s1c[i % len(s1c)], s2c[i % len(s2c)]) for i in idx])
    fold["chosen"] = chosen
    fold["diff"] = {"point": fold["rows"][chosen]["auc"] - fold["rows"]["classical"]["auc"],
                    "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}
    out["folds"].append(fold)
    r = fold["rows"]
    print(f"held {fold['held']}: alpha {fold['hill_alpha_pooled']:.2f} chosen {chosen} | "
          + " ".join(f"{k}={r[k]['auc']:.3f}" for k in r)
          + f" | diff {fold['diff']['point']:+.3f} [{fold['diff']['ci_lo']:+.3f},{fold['diff']['ci_hi']:+.3f}]")


def file_level(key_fn):
    d = np.array([key_fn(f) for f in out["folds"]])
    rb = np.array([rng.choice(d, size=len(d)).mean() for _ in range(NBOOT_FILES)])
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.percentile(rb, 2.5)), "ci_hi": float(np.percentile(rb, 97.5)),
            "n_positive": int((d > 0).sum()), "min": float(d.min()), "max": float(d.max())}


out["recording_level"] = {"chosen_vs_classical": file_level(lambda f: f["diff"]["point"])}
for k in CANDIDATES:
    out["recording_level"][f"{k}_vs_classical"] = file_level(lambda f, k=k: f["rows"][k]["auc"] - f["rows"]["classical"]["auc"])
alphas = [f["hill_alpha_pooled"] for f in out["folds"]]
even_share = float(np.mean([f["rows"]["M"]["parity"]["J_even"] / f["rows"]["M"]["parity"]["J_full"] for f in out["folds"]]))
rl = out["recording_level"]["chosen_vs_classical"]
out["alpha_hat"] = {"min": float(min(alphas)), "max": float(max(alphas)), "mean": float(np.mean(alphas))}
out["even_share_M"] = even_share
out["mean_auc_chosen"] = float(np.mean([f["rows"][f["chosen"]]["auc"] for f in out["folds"]]))
out["mean_auc_classical"] = float(np.mean([f["rows"]["classical"]["auc"] for f in out["folds"]]))
out["criteria"] = {"1_mechanism": bool(out["alpha_hat"]["mean"] < 4 or even_share > 0.5),
                   "2w_window_ci_all_folds": bool(all(f["diff"]["ci_lo"] > 0 for f in out["folds"])),
                   "2w_n_folds_ci_positive": int(sum(f["diff"]["ci_lo"] > 0 for f in out["folds"])),
                   "2r_recording_ci_excludes_zero": bool(rl["ci_lo"] > 0),
                   "3_nontrivial": bool(out["mean_auc_chosen"] >= 0.60)}
c = out["criteria"]
out["PASS"] = bool(c["1_mechanism"] and c["2r_recording_ci_excludes_zero"] and c["3_nontrivial"])
print("recording level chosen vs classical:", rl)
print("criteria:", c, "PASS", out["PASS"])
print("written", write_result("l07", out))
