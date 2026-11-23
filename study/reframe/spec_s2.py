"""mc_kernel spec for supplement S2 (design and registered criteria: DESIGN_S2.md).

Run from study/reframe:
    python3 mc_kernel.py spec_s2.py --resume
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1v4 as H  # noqa: E402
import h1v4_s2 as S  # noqa: E402

RUN_ID = "h1v4-s2-2026-11-22"

CONTROLS = [
    {"name": "C6 window MC vs exact CF (a* = 4)",
     "params": {"block": "control", "ctl": "C6"}, "expect": {"z100": (0.0, 4.0), "z1000": (0.0, 4.0)}},
    {"name": "C7 t3 sampler, tail constant and moments",
     "params": {"block": "control", "ctl": "C7"},
     "expect": {"maxz_tail": (0.0, 4.0), "c0_relerr": (0.0, 1e-3), "maxz_mom": (0.0, 4.0)}},
    {"name": "C8 RflyMAD pools and flight split",
     "params": {"block": "control", "ctl": "C8"},
     "expect": {"n_flights": (29.0, 0.0), "split_repeat": (1.0, 0.0), "sd_ratio": (2.25, 0.75)}},
    {"name": "C9 winsorizer uses training data only",
     "params": {"block": "control", "ctl": "C9"}, "expect": {"same_thr": (1.0, 0.0), "finite": (1.0, 0.0)}},
    {"name": "C10 LePage prediction vs symmetric stable CDF",
     "params": {"block": "control", "ctl": "C10"}, "expect": {"maxerr": (0.0, 0.005)}},
]

TE = {"sym": (0.2, 0.2), "one": (0.4, 0.0)}

GRID = []
for n in (50, 200, 1000):  # S2-A
    for d in ("oo", "oe", "ee", "eo"):
        GRID.append({"block": "X", "src": "gauss", "cont": "pareto", "dict": d, "astar": 8.0, "n": n, "te": "sym"})
    for d in ("oe", "ee"):
        GRID.append({"block": "X", "src": "gauss", "cont": "pareto", "dict": d, "astar": 8.0, "n": n, "te": "one"})
for d in ("oe", "ee"):  # S2-B crossover
    for te in ("sym", "one"):
        GRID.append({"block": "X", "src": "gauss", "cont": "t3", "dict": d, "astar": 8.0, "n": 200, "te": te})
GRID += [{"block": "T", "dict": d} for d in ("ee", "oe")]  # S2-B training
for n in (200, 1000):  # S2-D
    for d in ("oe", "ee"):
        for te in ("sym", "one"):
            GRID.append({"block": "X", "src": "rfly", "cont": "pareto", "dict": d, "astar": 8.0, "n": n, "te": te})


def _control(ctl: str) -> dict:
    if ctl == "C6":
        cols = H.columns("ee", 4.0)
        rng = np.random.default_rng(H.seed_of({"ctl": "C6"}))
        K, _ = H.fit(cols, H.draw(rng, 200, 0.05, H.G_PLUS, 0.5), H.draw(rng, 200, 0.05, H.G_MINUS, 0.5))
        E = H.CFEngine(cols, K)
        cf = E.auc((0.05, 0.05), 0.5, [100, 1000])
        cl = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))
        M = 4000
        a = S.window_auc(rng, cols, K, K, K[:-1], (-np.inf, np.inf), (0.05, 0.05), cl,
                         S.Contaminant("pareto"), Ns=np.array([100, 1000]), M=M)["K"]
        out = {}
        for i, N in enumerate((100, 1000)):
            p = a[i]
            out[f"cf{N}"], out[f"mc{N}"] = float(cf[i]), float(p)
            out[f"z{N}"] = float((cf[i] - p) / math.sqrt(max(p * (1 - p), 1e-4) / M))
        return out
    if ctl == "C7":
        rng = np.random.default_rng(H.seed_of({"ctl": "C7"}))
        c = S.Contaminant("t3")
        x = c.sample(rng, 4_000_000)
        zt = []
        for q in (10.0, 20.0):
            p_emp = float(np.mean(np.abs(x) > q))
            p_ex = 2 * stats.t.sf(q, 3)
            zt.append(abs(p_emp - p_ex) / math.sqrt(p_ex * (1 - p_ex) / x.size))
        c0_ex = 1e9 * 2 * stats.t.sf(1e3, 3)
        zm = []
        for q in (0.5, 1.2, 2.0):
            v = np.abs(x) ** q
            zm.append(abs(v.mean() - c.moment("e", q)) / (v.std() / math.sqrt(v.size)))
        return {"maxz_tail": float(max(zt)), "c0_exact_x1e3": float(c0_ex),
                "c0_relerr": float(abs(c0_ex - S.T3_C0) / S.T3_C0), "maxz_mom": float(max(zm))}
    if ctl == "C8":
        seed = H.seed_of({"rfly_split": 1})
        P = S.rfly_pools(seed)
        S.rfly_pools.cache_clear()
        Q = S.rfly_pools(seed)
        hi, lo = P["train"]
        return {"n_flights": float(P["n_flights"]), "split_repeat": float(P["train_ids"] == Q["train_ids"]),
                "sd_ratio": float(np.std(hi.v) / np.std(lo.v)),
                "sizes": [int(p.v.size) for p in P["train"] + P["test"]], "cache_sha256": P["cache_sha256"]}
    if ctl == "C9":
        rng = np.random.default_rng(H.seed_of({"ctl": "C9"}))
        cols = H.columns("ee", 8.0)
        cl = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))
        cont = S.Contaminant("pareto")
        xp, xm = S.draw(rng, 200, 0.05, cl[0], cont), S.draw(rng, 200, 0.05, cl[1], cont)
        K, KW, KS, thr, _ = S.fit_all(cols, xp, xm)
        _ = S.draw(rng, (20, 1000), 0.2, cl[0], cont)  # unrelated test windows
        thr2 = S.thresholds(xp, xm)
        return {"same_thr": float(thr == thr2),
                "finite": float(all(np.all(np.isfinite(v)) for v in (K, KW, KS)))}
    if ctl == "C10":
        gam = 3.0 / 8.0
        Z = S.lepage(gam, 0.0, 200_000, np.random.default_rng(H.seed_of({"ctl": "C10"})))
        errs = [abs(float(np.mean(Z <= x)) - H.stable_cdf_sym(x, gam)) for x in (-2.0, -0.5, 1.0)]
        return {"maxerr": float(max(errs))}
    raise ValueError(ctl)


def cell(params: dict, memo: dict) -> dict:
    if params["block"] == "control":
        return _control(params["ctl"])
    seed = H.seed_of(params)
    if params["block"] == "X":
        return S.crossover_cell(params["dict"], params["astar"], params["n"], TE[params["te"]],
                                params["cont"], params["src"], seed)
    if params["block"] == "T":
        return S.train_block_t3(params["dict"], seed)
    raise ValueError(params)
