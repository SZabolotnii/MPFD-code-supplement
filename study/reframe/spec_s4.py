"""mc_kernel spec for supplement S4 (design and registered criteria: DESIGN_S4.md).

Run from study/reframe:
    python3 mc_kernel.py spec_s4.py --resume
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1v4 as H  # noqa: E402
import h1v4_s4 as F  # noqa: E402
import spec_s2  # noqa: E402
import spec_s3  # noqa: E402

RUN_ID = "h1v4-s4-2026-11-24"
STORED = {2: HERE / "results" / "h1v4-s2-2026-11-22", 3: HERE / "results" / "h1v4-s3-2026-11-23"}
REPRO_KEYS = ("medPeak", "onFinite", "medNon", "medNstar", "ratio", "medAucEnd", "predErr",
              "medAucWEnd", "medAucSEnd", "lock", "pos", "medCond")
EPS_TR = (0.05, 0.05)

# ----------------------------------------------------------------------------- the grid
# D: every crossover cell of runs 2 and 3, same params, same seeds
D_CELLS = []
for p in spec_s2.GRID:
    if p["block"] == "X":
        D_CELLS.append({"block": "D", "run": 2, "oblock": "X", **{k: v for k, v in p.items() if k != "block"}})
for p in spec_s3.GRID:
    if p["block"] in ("A", "B"):
        D_CELLS.append({"block": "D", "run": 3, "oblock": p["block"],
                        **{k: v for k, v in p.items() if k != "block"}})

# A: registered from results/design_probe_s4.json by the rule of DESIGN_S4.md
TE_A = {"sym05": (0.05, 0.05), "sym20": (0.2, 0.2), "one10": (0.1, 0.0), "one40": (0.4, 0.0)}
A_CELLS = [{"block": "A4", "dict": d, "alpha": al, "n": n, "te": te} for d, al, n, te in (
    ("oo3", 2.2, 200, "sym20"), ("oe3", 2.2, 200, "sym05"), ("oe3", 2.2, 200, "one10"),
    ("oe3", 2.2, 1000, "sym20"), ("oe3", 2.2, 1000, "one40"), ("oo3", 2.5, 200, "sym20"),
    ("oe3", 2.5, 1000, "sym20"), ("oe3", 2.5, 1000, "one40"),
    ("poly3", 2.2, 200, "sym20"), ("poly3", 2.5, 200, "sym20"))]  # last two: |I| = 2, reported

TE_B = {"in+": (0.075, 0.025), "in-": (0.025, 0.075)}
B_CELLS = [{"block": "B4", "te": te} for te in TE_B]

TE_C = {"sym": (0.05, 0.05), "one": (0.10, 0.0)}
C_CELLS = [{"block": "C4", "dict": d, "astar": a, "alpha": al, "n": n, "te": te} for d, a, al, n, te in (
    ("oe", 3.0, 1.2, 200, "sym"), ("oe", 3.0, 1.2, 200, "one"), ("ee", 3.0, 1.2, 200, "sym"),
    ("ee", 3.0, 1.2, 200, "one"), ("ee", 3.0, 1.2, 1000, "sym"), ("oe", 4.0, 1.2, 200, "sym"),
    ("oe", 4.0, 1.2, 200, "one"), ("oe", 4.0, 1.2, 1000, "sym"), ("oe", 4.0, 1.2, 1000, "one"),
    ("ee", 4.0, 1.2, 200, "sym"), ("ee", 4.0, 1.2, 200, "one"), ("ee", 4.0, 1.2, 1000, "sym"),
    ("ee", 4.0, 1.2, 1000, "one"), ("ee", 3.0, 1.5, 200, "sym"), ("oe", 4.0, 1.5, 200, "sym"),
    ("oe", 4.0, 1.5, 200, "one"), ("ee", 4.0, 1.5, 200, "sym"), ("ee", 4.0, 1.5, 200, "one"))]

TE_E = {"fwd10": (0.1, 0.0), "rev10": (0.0, 0.1), "fwd40": (0.4, 0.0), "rev40": (0.0, 0.4)}
E_CELLS = [{"block": "E4", "te": te} for te in TE_E]

GRID = A_CELLS + B_CELLS + C_CELLS + E_CELLS + D_CELLS

CONTROLS = [
    {"name": "C14 the S4 summary reproduces a stored run-3 cell (same seeds)",
     "params": {"block": "control", "ctl": "C14"}, "expect": {"maxdiff": (0.0, 1e-9)}},
    {"name": "C15 the S4 summary reproduces a stored run-2 RflyMAD cell (same seeds)",
     "params": {"block": "control", "ctl": "C15"}, "expect": {"maxdiff": (0.0, 1e-9)}},
]


# ----------------------------------------------------------------------------- helpers

def stored_values(run: int, orig: dict) -> dict:
    for f in sorted(STORED[run].glob("*.json")):
        if f.name == "RUN.json":
            continue
        d = json.loads(f.read_text())
        if d.get("params") == orig:
            return d["values"]
    raise KeyError(f"no stored run-{run} cell for {orig}")


def repro_diff(new: dict, old: dict) -> float:
    d = 0.0
    for k in REPRO_KEYS:
        if k not in old:
            continue
        a, b = float(new[k]), float(old[k])
        if math.isnan(a) and math.isnan(b):
            continue
        d = max(d, abs(a - b)) if not (math.isnan(a) or math.isnan(b)) else math.inf
    return d


def rerun(p: dict) -> dict:
    orig = {"block": p["oblock"], **{k: v for k, v in p.items() if k not in ("block", "run", "oblock")}}
    if p["run"] == 3:
        te = (spec_s3.TE_A if p["oblock"] == "A" else spec_s3.TE_B)[p["te"]]
        out = F.rerun_s3(orig, spec_s3.EPS_TR, te)
    else:
        out = F.rerun_s2(orig, spec_s2.TE[p["te"]])
    out["reproMaxDiff"] = repro_diff(out, stored_values(p["run"], orig))
    return out


def _control(ctl: str) -> dict:
    if ctl == "C14":
        out = rerun({"block": "D", "run": 3, "oblock": "A", "dict": "oo", "alpha": 1.2, "astar": 3.0,
                     "n": 200, "te": "sym"})
        return {"maxdiff": out["reproMaxDiff"]}
    if ctl == "C15":
        out = rerun({"block": "D", "run": 2, "oblock": "X", "src": "rfly", "cont": "pareto", "dict": "ee",
                     "astar": 8.0, "n": 200, "te": "sym"})
        return {"maxdiff": out["reproMaxDiff"]}
    raise ValueError(ctl)


def cell(params: dict, memo: dict) -> dict:
    b = params["block"]
    if b == "control":
        return _control(params["ctl"])
    if b == "D":
        return rerun(params)
    seed = H.seed_of(params)
    if b == "A4":
        return F.new_cell(params["dict"], 3.0, params["alpha"], params["n"], EPS_TR, TE_A[params["te"]],
                          "gauss", seed)
    if b == "B4":
        return F.new_cell("ee", 4.0, 1.2, 1000, EPS_TR, TE_B[params["te"]], "gauss", seed)
    if b == "C4":
        return F.new_cell(params["dict"], params["astar"], params["alpha"], params["n"], EPS_TR,
                          TE_C[params["te"]], "rfly", seed)
    if b == "E4":
        return F.new_cell("ee", 8.0, 3.0, 200, EPS_TR, TE_E[params["te"]], "scale", seed)
    raise ValueError(params)
