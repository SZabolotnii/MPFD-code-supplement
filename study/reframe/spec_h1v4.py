"""mc_kernel spec for the stage-6 study of hypothesis H1 v4 (design and registered criteria: DESIGN.md).

Run from study/reframe:
    python3 mc_kernel.py spec_h1v4.py --resume
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import levy_stable, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1v4 as H  # noqa: E402

RUN_ID = "h1v4-2026-11-22"

CONTROLS = [
    {"name": "C1 CF engine vs normal formula (clean windows)",
     "params": {"block": "control", "ctl": "C1"}, "expect": {"err": (0.0, 1e-6)}},
    {"name": "C2 CF engine vs brute-force MC (contaminated windows)",
     "params": {"block": "control", "ctl": "C2"}, "expect": {"z50": (0.0, 4.0), "z2000": (0.0, 4.0)}},
    {"name": "C3 stable CDF vs scipy levy_stable",
     "params": {"block": "control", "ctl": "C3"}, "expect": {"maxerr": (0.0, 1e-3)}},
    {"name": "C4 H sampler: same parity positive, opposite symmetric fair",
     "params": {"block": "control", "ctl": "C4"},
     "expect": {"pos_same": (1.0, 0.0), "pos_opp": (0.5, 0.015)}},
    {"name": "C5 population moments vs 4e6-draw sample",
     "params": {"block": "control", "ctl": "C5"}, "expect": {"maxz": (0.0, 4.0)}},
]

SPLITS = {"bal": (0.05, 0.05), "plus": (0.10, 0.0), "minus": (0.0, 0.10)}

GRID = [{"block": "A", "dict": d, "astar": a, "pplus": p, "split": s}
        for d in ("oo", "oe", "ee", "eo") for a in (2.4, 4.0) for p in (0.5, 0.8) for s in SPLITS]
GRID += [{"block": "B", "dict": d} for d in ("oo", "oe", "ee", "eo")]


def _control(ctl: str) -> dict:
    if ctl == "C1":
        cols = H.columns("oo", 4.0)
        E = H.CFEngine(cols, np.array([0.0, 1.0, 0.0]), pareto=False)
        got = E.auc_curve(E.logcf((0.0, 0.0), 0.5), [20])[0]
        want = norm.cdf(20 * 0.3 / math.sqrt(20 * (1.3 ** 2 + 1.0)))
        return {"auc": float(got), "err": float(got - want)}
    if ctl == "C2":
        cols = H.columns("oo", 4.0)
        k = np.array([0.1, 0.3, -0.05])
        E = H.CFEngine(cols, k)
        cf = E.auc_curve(E.logcf((0.05, 0.05), 0.5), [50, 2000])
        rng = np.random.default_rng(H.seed_of({"ctl": "C2"}))
        out = {}
        for N, P, c in ((50, 200_000, cf[0]), (2000, 20_000, cf[1])):
            wins = 0
            for _ in range(P // 1000):
                xp = rng.permutation(H.draw(rng, 1000 * N, 0.05, H.G_PLUS, 0.5))
                xm = rng.permutation(H.draw(rng, 1000 * N, 0.05, H.G_MINUS, 0.5))
                sp = (H.phi(cols, xp) @ k).reshape(1000, N).sum(1)
                sm = (H.phi(cols, xm) @ k).reshape(1000, N).sum(1)
                wins += int((sp > sm).sum())
            p = wins / P
            out[f"cf{N}"], out[f"mc{N}"] = float(c), p
            out[f"z{N}"] = float((c - p) / math.sqrt(p * (1 - p) / P))
        return out
    if ctl == "C3":
        sc = H.stable_scale_pow(0.75) ** (1 / 0.75)
        errs = [abs(H.stable_cdf_sym(x, 0.75) - levy_stable.cdf(x, 0.75, 0.0, scale=sc))
                for x in (-5.0, -2.0, -0.5, 1.0)]
        return {"maxerr": float(max(errs)), "lambda_m": float(H.lambda_mid(0.75, 0.75))}
    if ctl == "C4":
        rng = np.random.default_rng(H.seed_of({"ctl": "C4"}))
        hs = H.h_sample(H.columns("oo", 4.0), 0.5, 20000, rng)
        ho = H.h_sample(H.columns("oe", 4.0), 0.5, 20000, rng)
        return {"pos_same": float(np.mean(hs > 0)), "pos_opp": float(np.mean(ho > 0))}
    if ctl == "C5":
        rng = np.random.default_rng(H.seed_of({"ctl": "C5"}))
        zs = []
        for name in ("oo", "ee"):
            L = H.columns(name, 4.0)[:-1]
            for eps, clean in ((0.05, H.G_PLUS), (0.10, H.G_MINUS)):
                pop = H.population_mean(L, eps, clean, 0.8)
                P = H.phi(L, H.draw(rng, 4_000_000, eps, clean, 0.8))
                zs.extend(np.abs(P.mean(0) - pop) / (P.std(0) / math.sqrt(P.shape[0])))
        return {"maxz": float(max(zs))}
    raise ValueError(ctl)


def cell(params: dict, memo: dict) -> dict:
    if params["block"] == "control":
        return _control(params["ctl"])
    seed = H.seed_of(params)
    if params["block"] == "A":
        return H.block_a(params["dict"], params["astar"], params["pplus"], SPLITS[params["split"]],
                         seed, memo)
    if params["block"] == "B":
        return H.block_b(params["dict"], seed)
    raise ValueError(params)
