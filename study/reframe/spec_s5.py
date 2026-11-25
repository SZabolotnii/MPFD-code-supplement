"""mc_kernel spec for supplement S5 (design and registered criteria: DESIGN_S5.md).

Run from study/reframe:
    python3 mc_kernel.py spec_s5.py --resume
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1v4 as H  # noqa: E402
import h1v4_s5 as G  # noqa: E402

RUN_ID = "h1v4-s5-2026-11-24"
STORED_S3 = HERE / "results" / "h1v4-s3-2026-11-23"
EPS_TR = (0.05, 0.05)

# ----------------------------------------------------------------------------- the grid
# F: the limit law of Cor. detwindow; kappa_0 by the registered rule (DESIGN_S5.md)
F_CELLS = [{"block": "F5", "dict": d, "n": n} for d in ("oo4", "oe4") for n in (32000, 128000, 512000)]

# G: registered from results/design_probe_s5.json by the rule of DESIGN_S5.md
TE_G = {"fwd40": (0.4, 0.0), "rev20": (0.0, 0.2), "rev40": (0.0, 0.4)}
G_CELLS = [{"block": "G5", "src": s, "alpha": 2.2, "n": n, "te": te, "role": role} for s, n, te, role in (
    ("gauss", 200, "rev40", "collapse"), ("mirror", 200, "fwd40", "collapse"), ("mirror", 200, "rev40", "collapse"),
    ("gauss", 200, "rev20", "drift"), ("gauss", 1000, "rev40", "drift"))]

GRID = F_CELLS + G_CELLS

CONTROLS = [
    {"name": "C16 the S5 fits reproduce the stored S3-C median |K_t| (same seeds, all three n, both dictionaries)",
     "params": {"block": "control", "ctl": "C16"}, "expect": {"maxreldiff": (0.0, 1e-12)}},
    {"name": "C17 CFEngineClean with the run-1 clean laws equals the run-1 CFEngine",
     "params": {"block": "control", "ctl": "C17"}, "expect": {"maxdiff": (0.0, 1e-12)}},
    {"name": "C18 the compensated LePage sum matches the Gil-Pelaez CDF of Z_{gamma,+-1}",
     "params": {"block": "control", "ctl": "C18"}, "expect": {"maxdiff": (0.0, 0.003)}},
]


# ----------------------------------------------------------------------------- helpers

def stored_s3c(name: str) -> dict:
    for f in sorted(STORED_S3.glob("*.json")):
        if f.name == "RUN.json":
            continue
        d = json.loads(f.read_text())
        if d.get("params") == {"block": "C", "dict": name}:
            return d["values"]
    raise KeyError(name)


def kappa0(name: str, memo: dict) -> float:
    key = f"kappa0_{name}"
    if key not in memo:
        memo[key] = G.kappa0_rule(name)
    return memo[key]


def _control(ctl: str) -> dict:
    if ctl == "C16":
        d = 0.0
        for name in G.F_DICTS:
            old = stored_s3c(name)
            for n in G.F_NS[-3:]:
                new = float(np.median(np.abs(G.f_fits(name, n)[:, -1])))
                d = max(d, abs(new - old[f"medKt_n{n}"]) / old[f"medKt_n{n}"])
        return {"maxreldiff": d}
    if ctl == "C17":
        cols = H.columns("oo", 4.0)
        k = np.array([0.4, 1.0, -2e-4])
        a = H.CFEngine(cols, k)
        b = G.CFEngineClean(cols, k, (H.G_PLUS, H.G_MINUS))
        Ns = [100, 10 ** 6]
        d = float(np.max(np.abs(a.auc((0.05, 0.05), 0.5, Ns) - b.auc((0.05, 0.05), 0.5, Ns))))
        return {"maxdiff": d}
    if ctl == "C18":
        d = 0.0
        for gam in (2.2 / 3, 0.75):
            for beta in (1.0, -1.0):
                Z = np.sort(G.lepage_c(gam, beta, 200_000, np.random.default_rng(H.seed_of({"C18": gam, "b": beta}))))
                for x in ((0.05, 0.1, 0.3, 0.6, 1.0, 3.0) if beta > 0 else (-5.0, -1.0, -0.3, 0.1, 0.3, 1.0)):
                    d = max(d, abs(G.stable_cdf(x, gam, beta) - np.searchsorted(Z, x) / Z.size))
        return {"maxdiff": float(d)}
    raise ValueError(ctl)


def _flat(out: dict) -> dict:
    flat = {}
    for k, v in out.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}_{kk}"] = vv
        else:
            flat[k] = v
    return flat


def cell(params: dict, memo: dict) -> dict:
    b = params["block"]
    if b == "control":
        return _control(params["ctl"])
    if b == "F5":
        return _flat(G.f_cell(params["dict"], params["n"], kappa0(params["dict"], memo)))
    if b == "G5":
        return G.g_cell(params["alpha"], params["n"], TE_G[params["te"]], params["src"], H.seed_of(params), EPS_TR)
    raise ValueError(params)
