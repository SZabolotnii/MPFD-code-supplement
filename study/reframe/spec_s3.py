"""mc_kernel spec for supplement S3 (design and registered criteria: DESIGN_S3.md).

Run from study/reframe:
    python3 mc_kernel.py spec_s3.py --resume
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1v4 as H  # noqa: E402
import h1v4_s2 as S  # noqa: E402
import h1v4_s3 as T  # noqa: E402

RUN_ID = "h1v4-s3-2026-11-23"

CONTROLS = [
    {"name": "C11 Pareto(alpha) sampler, tails and moments",
     "params": {"block": "control", "ctl": "C11"}, "expect": {"maxz": (0.0, 4.0)}},
    {"name": "C12 stable CDF vs LePage at the new gamma",
     "params": {"block": "control", "ctl": "C12"}, "expect": {"maxerr": (0.0, 0.005)}},
    {"name": "C13 S3 engine reproduces run 2 at alpha_c = 3",
     "params": {"block": "control", "ctl": "C13"}, "expect": {"maxdiff": (0.0, 1e-12)}},
]

TE_A = {"sym": (0.05, 0.05), "one": (0.10, 0.0)}
TE_B = {"in+": (0.3, 0.1), "in-": (0.1, 0.3), "rev10": (0.0, 0.1), "rev40": (0.0, 0.4)}
EPS_TR = (0.05, 0.05)

GRID = []
for alpha in (1.2, 1.5):  # S3-A
    for astar in (3.0, 4.0):
        ns = (200, 1000) if (alpha, astar) == (1.2, 4.0) else (200,)
        for n in ns:
            for d in ("oo", "oe", "ee", "eo"):
                GRID.append({"block": "A", "dict": d, "alpha": alpha, "astar": astar, "n": n, "te": "sym"})
            for d in ("oe", "ee"):
                GRID.append({"block": "A", "dict": d, "alpha": alpha, "astar": astar, "n": n, "te": "one"})
for d in ("oe", "ee"):  # S3-B
    for te in TE_B:
        GRID.append({"block": "B", "dict": d, "alpha": 3.0, "astar": 8.0, "n": 200, "te": te})
GRID += [{"block": "C", "dict": d} for d in T.RATE_DICTS]  # S3-C


def _control(ctl: str) -> dict:
    if ctl == "C11":
        zs = []
        for alpha in (1.2, 1.5):
            c = T.Pareto(alpha)
            x = c.sample(np.random.default_rng(H.seed_of({"ctl": "C11", "a": alpha})), 4_000_000)
            for q in (10.0, 100.0):
                p_emp, p_ex = float(np.mean(np.abs(x) > q)), q ** -alpha
                zs.append(abs(p_emp - p_ex) / math.sqrt(p_ex * (1 - p_ex) / x.size))
            for q in (0.2, 0.5):
                v = np.abs(x) ** q
                zs.append(abs(v.mean() - c.moment("e", q)) / (v.std() / math.sqrt(v.size)))
        return {"maxz": float(max(zs))}
    if ctl == "C12":
        errs = []
        for gam in (0.3, 0.375, 0.4, 0.5):
            Z = S.lepage(gam, 0.0, 200_000, np.random.default_rng(H.seed_of({"ctl": "C12", "g": gam})))
            errs += [abs(float(np.mean(Z <= x)) - H.stable_cdf_sym(x, gam)) for x in (-2.0, -0.5, 1.0)]
        return {"maxerr": float(max(errs))}
    if ctl == "C13":
        seed = H.seed_of({"ctl": "C13"})
        a = T.crossover_rep(("ee", 8.0, 3.0, 200, EPS_TR, (0.2, 0.2), seed))
        b = S.crossover_rep(("ee", 8.0, 200, (0.2, 0.2), "pareto", "gauss", seed))
        keys = ("Kt", "Nstar", "N_on", "auc_end", "aucW_end", "aucS_end")
        d = [abs(a[k] - b[k]) if not (math.isnan(a[k]) and math.isnan(b[k])) else 0.0 for k in keys]
        return {"maxdiff": float(max(d)), **{f"s3_{k}": a[k] for k in keys}}
    raise ValueError(ctl)


def cell(params: dict, memo: dict) -> dict:
    if params["block"] == "control":
        return _control(params["ctl"])
    seed = H.seed_of(params)
    if params["block"] in ("A", "B"):
        te = (TE_A if params["block"] == "A" else TE_B)[params["te"]]
        return T.crossover_cell(params["dict"], params["astar"], params["alpha"], params["n"], EPS_TR, te, seed)
    if params["block"] == "C":
        return T.rate_block(params["dict"], seed)
    raise ValueError(params)
