"""Evaluate the registered criteria D0-D3, A1-A2, B1, C1-C3, E1 of DESIGN_S4.md on the hashed S4
artifacts.

Reads results/<run-id>/*.json only (no recomputation). Usage: python criteria_s4.py [run-id]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "h1v4-s4-2026-11-24"
D = Path(__file__).resolve().parent / "results" / RUN


def load(block: str) -> list[dict]:
    rows = []
    for f in sorted(D.glob("*.json")):
        if f.name == "RUN.json":
            continue
        a = json.loads(f.read_text())
        if a["params"].get("block") == block:
            rows.append(a)
    return rows


def tag(p: dict) -> str:
    keys = [k for k in ("run", "oblock", "src", "cont", "dict", "alpha", "astar", "n", "te") if k in p]
    return p["block"] + " " + " ".join(f"{k}={p[k]}" for k in keys)


def in_d1(p: dict) -> bool:
    """Q2 (run 2: symmetric, synthetic Gaussian clean laws, Pareto or t3) and R2 (run 3: symmetric S3-A)."""
    if p["run"] == 2:
        return p["te"] == "sym" and p["src"] == "gauss"
    return p["oblock"] == "A" and p["te"] == "sym"


def in_d2(p: dict) -> bool:
    """Q3 (run 2: one-sided) and R3 (run 3: one-sided S3-A and all of S3-B)."""
    if p["run"] == 2:
        return p["te"] == "one"
    return p["oblock"] == "B" or p["te"] == "one"


def r1(v) -> bool:
    return v["medPeak"] >= 0.80 and v["onFinite"] >= 0.80 and v["medNon"] <= 2e4


def r2(v) -> bool:
    return math.isfinite(v["ratio"]) and 0.67 <= v["ratio"] <= 1.5


def row(p, v) -> str:
    rq, eq = v.get("ratioQ", [math.nan] * 4), v.get("errQ", [math.nan] * 4)
    return (f'{tag(p):62s} | peak {v["medPeak"]:.3f} onF {v["onFinite"]:.2f} N* {v.get("medNstar", math.nan):8.4g} '
            f'ratio {v.get("ratio", math.nan):.3f} [{rq[0]:.2f},{rq[2]:.2f}] | end {v["medAucEnd"]:.3f} '
            f'pred {v.get("medPred", math.nan):.3f} err {v.get("predErr", math.nan):.3f} p90 {eq[3]:.3f} | '
            f'fw {v.get("fwMaxDiff", math.nan):.3f} | W {v["medAucWEnd"]:.3f} | lock {v.get("lock", math.nan):.2f} '
            f'pos {v.get("pos", math.nan):.2f}')


def main() -> int:
    crit = ("D0", "D1", "D2", "D3", "A1", "A2", "B1", "C1", "C2", "C3", "E1")
    fails = {k: [] for k in crit}
    tot = {k: 0 for k in crit}

    print("== D: runs 2 and 3 re-run with per-fit values")
    for a in sorted(load("D"), key=lambda r: tag(r["params"])):
        p, v = a["params"], a["values"]
        print(row(p, v) + f' | repro {v["reproMaxDiff"]:.1e}')
        tot["D0"] += 1
        if not v["reproMaxDiff"] <= 1e-9:
            fails["D0"].append(tag(p))
        if in_d1(p):
            tot["D1"] += 1
            tot["D3"] += 1
            q = v["ratioQ"]
            if not (q[0] >= 0.5 and q[2] <= 2.0):
                fails["D1"].append(f'{tag(p)} [{q[0]:.2f}, {q[2]:.2f}]')
            if not v["fwMaxDiff"] <= 0.05:
                fails["D3"].append(f'{tag(p)} {v["fwMaxDiff"]:.3f}')
        if in_d2(p):
            tot["D2"] += 1
            if not v["errQ"][3] <= 0.15:
                fails["D2"].append(f'{tag(p)} p90 {v["errQ"][3]:.3f}')

    print("\n== A: (x, x^3), (x, |x|^3), (x, x^2, x^3) at alpha_c in (2, 3)")
    for a in load("A4"):
        p, v = a["params"], a["values"]
        if p["dict"] == "poly3":
            print(f'{tag(p):62s} | peak {v["medPeak"]:.3f} onF {v["onFinite"]:.2f} medNon {v["medNon"]:.4g} '
                  f'NonQ [{v["NonQ"][0]:.4g},{v["NonQ"][2]:.4g}] end {v["medAucEnd"]:.3f} W {v["medAucWEnd"]:.3f} '
                  f'P(K2>0) {v["pos_K2"]:.2f} P(K3>0) {v["pos_K3"]:.2f}  [reported]')
            continue
        print(row(p, v))
        if p["te"].startswith("sym"):
            tot["A1"] += 1
            if not (r1(v) and r2(v)):
                fails["A1"].append(tag(p))
        else:
            tot["A2"] += 1
            if not v["predErr"] <= 0.10:
                fails["A2"].append(tag(p))

    print("\n== B: locked interior beta")
    for a in load("B4"):
        p, v = a["params"], a["values"]
        print(row(p, v) + f' beta {v["beta_te"]:+.2f}')
        tot["B1"] += 1
        side = v["medAucEnd"] < 0.5 if v["beta_te"] > 0 else v["medAucEnd"] > 0.5
        if not (v["lock"] >= 0.95 and v["predErr"] <= 0.10 and side):
            fails["B1"].append(tag(p))

    print("\n== C: RflyMAD pools, run-3 regime")
    for a in sorted(load("C4"), key=lambda r: tag(r["params"])):
        p, v = a["params"], a["values"]
        print(row(p, v) + f' Ksub* {v["ksub_lstar"]:+.4f} cost10 {v["medCostW10"]:+.3f}')
        if p["te"] == "sym":
            tot["C1"] += 1
            tot["C3"] += 1
            if not (r1(v) and r2(v)):
                fails["C1"].append(tag(p))
            if not (v["medAucWEnd"] >= 0.90 and v["medCostW10"] >= -0.02):
                fails["C3"].append(tag(p))
        else:
            tot["C2"] += 1
            if not v["predErr"] <= 0.10:
                fails["C2"].append(tag(p))

    print("\n== E: scale-only clean laws")
    for a in load("E4"):
        p, v = a["params"], a["values"]
        print(row(p, v) + f' beta {v["beta_te"]:+.2f} mu_te_L {v["mu_te_L"]:+.4f}')
        if p["te"] == "rev40":
            continue
        tot["E1"] += 1
        if not v["predErr"] <= 0.10:
            fails["E1"].append(tag(p))

    print("\nVerdicts")
    for k in crit:
        f = fails[k]
        need_all = k in ("D0", "A1", "A2", "B1", "C1", "C2", "C3", "E1")
        ok = (not f) if need_all else (tot[k] > 0 and (tot[k] - len(f)) / tot[k] >= 0.80)
        print(f'{k}: {"PASS" if ok else "FAIL"} ({tot[k] - len(f)}/{tot[k]})' + (f' — {"; ".join(f)}' if f else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
