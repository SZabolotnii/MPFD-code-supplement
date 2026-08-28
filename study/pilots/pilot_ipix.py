#!/usr/bin/env python3
"""Pilot P3: IPIX Dartmouth 1993 sea clutter, clutter-only cells (H0) vs primary target cell (H1),
same protocol and criteria as PILOTS_2026-08-28.md. Recording = file; leave-one-file-out
over all 14 target files. Scalar = I component of the like-polarised channel, first TX
polarisation. 8-bit ADC bytes are unsigned (int8 view wraps); centred per file. Writes
pilot_ipix.json beside itself."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # study/ -- these live in study/pilots/
from mcp import basis as B, eval as E, projection as P  # noqa: E402
from mcp.moments import hill_tail_index  # noqa: E402

STAGING = Path.home() / "Project/datasets/staging/ipix-dartmouth-1993"
SEED, NBOOT, N = 20260821, 200, 1024
TAUS = np.linspace(0.05, 0.99, 40); TAUS = TAUS[np.abs(TAUS - 0.5) > 0.04]
AGRID = np.round(np.arange(0.3, 1.91, 0.1), 2)
HGRID = np.round(np.linspace(0.55, 1.95, 8), 3)
# (file, primary cell, secondary cells) -- 1-indexed as on the IPIX datasets page
FILES = [("19931107_135603_starea", 9, range(8, 12)), ("19931107_141630_starea", 9, range(8, 12)),
         ("19931107_145028_starea", 8, range(7, 10)), ("19931108_213827_starea", 7, range(6, 9)),
         ("19931108_220902_starea", 7, range(6, 9)), ("19931109_191449_starea", 7, range(6, 9)),
         ("19931109_202217_starea", 7, range(6, 10)), ("19931110_001635_starea", 7, range(5, 9)),
         ("19931111_163625_starea", 8, range(7, 11)), ("19931118_023604_stareC0000", 8, range(7, 11)),
         ("19931118_035737_stareC0000", 10, range(8, 13)), ("19931118_162155_stareC0000", 7, range(6, 10)),
         ("19931118_162658_stareC0000", 7, range(6, 10)), ("19931118_174259_stareC0000", 7, range(6, 10))]


def load(name):
    f = netcdf_file(STAGING / f"{name}.cdf", "r", mmap=False)
    a = f.variables["adc_data"][:]                       # (nsweep, ntxpol, nrange, nadc), int8 view
    i = int(f.variables["adc_like_I"].data)
    x = a[:, 0, :, i].view(np.uint8).astype(float)      # (nsweep, nrange)
    x = x - 127.5
    return x


def windows(col):
    return col[: len(col) // N * N].reshape(-1, N)


recs = []
for name, prim, sec in FILES:
    x = load(name)
    clutter = [c for c in range(1, 15) if c not in sec]
    h0 = np.concatenate([windows(x[:, c - 1]) for c in clutter])
    h1 = windows(x[:, prim - 1])
    z0, z1 = (h0 - h0.mean()) / h0.std(), (h1 - h1.mean()) / h1.std()
    recs.append({"name": name, "h0": h0, "h1": h1, "clutter_cells": clutter, "primary": prim,
                 "stats": {"clutter": {"skew": float((z0 ** 3).mean()), "kurt": float((z0 ** 4).mean()), "sd": float(h0.std())},
                           "target": {"skew": float((z1 ** 3).mean()), "kurt": float((z1 ** 4).mean()), "sd": float(h1.std())}}})
    s = recs[-1]["stats"]
    print(f"{name}: clutter kurt {s['clutter']['kurt']:.2f} skew {s['clutter']['skew']:+.2f} sd {s['clutter']['sd']:.1f} | "
          f"target kurt {s['target']['kurt']:.2f} skew {s['target']['skew']:+.2f} sd {s['target']['sd']:.1f}")


def select(builder, grid, z1, z2):
    best = (-np.inf, None)
    for args in grid:
        cols = builder(*args)
        d = P.deflection(P.sample_moments(cols, z1), P.sample_moments(cols, z2))
        if d.cond <= P.COND_LIMIT and d.J > best[0]:
            best = (d.J, args)
    return best[1]


rng = np.random.default_rng(SEED)
out = {"N": N, "resampling_unit": "file (leave-one-file-out, 14 folds); window bootstrap within fold; recording-level paired CI over files",
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
    # both classes resampled by the smaller count so the paired resample is well defined
    diffs = np.array([E.auc(s1b[i % len(s1b)], s2b[i % len(s2b)]) - E.auc(s1c[i % len(s1c)], s2c[i % len(s2c)]) for i in idx])
    fold["best"] = best
    fold["diff"] = {"point": fold["rows"][best]["auc"] - fold["rows"]["classical"]["auc"],
                    "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5))}
    out["folds"].append(fold)
    r = fold["rows"]; p = r["M"]["parity"]
    print(f"held {fold['held']}: alpha {fold['hill_alpha_pooled']:.2f} n {fold['n_test']} M a*={astar[0]:.1f} even/full {p['J_even']/p['J_full']:.2f} | "
          + " ".join(f"{k}={r[k]['auc']:.3f}" for k in r) + f" | best {best} diff {fold['diff']['point']:+.3f} [{fold['diff']['ci_lo']:+.3f},{fold['diff']['ci_hi']:+.3f}]")
even_share = float(np.mean([f["rows"]["M"]["parity"]["J_even"] / f["rows"]["M"]["parity"]["J_full"] for f in out["folds"]]))
crit = {"1_mechanism": bool(np.mean([f["hill_alpha_pooled"] for f in out["folds"]]) < 4 or even_share > 0.5),
        "2_gain_over_control_all_folds": bool(all(f["diff"]["ci_lo"] > 0 for f in out["folds"])),
        "3_nontrivial": bool(np.mean([f["rows"][f["best"]]["auc"] for f in out["folds"]]) >= 0.60)}
out.update({"even_share_M": even_share, "criteria": crit, "PASS": all(crit.values())})
# recording-level summary: the file is the replicate
d = np.array([f["diff"]["point"] for f in out["folds"]])
rb = np.array([rng.choice(d, size=len(d)).mean() for _ in range(2000)])
out["recording_level"] = {"n_files": int(len(d)), "mean_diff": float(d.mean()), "ci_lo": float(np.percentile(rb, 2.5)),
                          "ci_hi": float(np.percentile(rb, 97.5)), "n_positive": int((d > 0).sum()),
                          "n_ci_excl_zero": int(sum(f["diff"]["ci_lo"] > 0 for f in out["folds"]))}
print("criteria", crit, "PASS", out["PASS"])
print("recording level:", out["recording_level"])
(HERE / "pilot_ipix.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
