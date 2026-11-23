"""Evaluate the registered criteria R1-R7 of DESIGN_S3.md on the hashed S3 artifacts.

Reads results/<run-id>/*.json only (no recomputation). Usage: python criteria_s3.py [run-id]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "h1v4-s3-2026-11-23"
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
    return f'{p["block"]} al={p["alpha"]} a*={p["astar"]:<3} {p["dict"]} n={p["n"]:<4d} {p["te"]}'


def main() -> int:
    X = sorted(load("A") + load("B"), key=lambda r: (r["params"]["block"], r["params"]["alpha"],
                                                     r["params"]["astar"], r["params"]["n"],
                                                     r["params"]["te"], r["params"]["dict"]))
    print("cell                                  | fail | peak  onFin medNon  medN*  ratio | end   pred  err   | endW  costW10 | lock  pos   beta")
    fails = {k: [] for k in ("R1", "R2", "R3", "R4a", "R4b", "R5", "R6", "R7")}
    tot = {k: 0 for k in fails}
    for a in X:
        p, v = a["params"], a["values"]
        print(f'{tag(p):37s} | {v["failed"]:.0f}    | {v["medPeak"]:.3f} {v["onFinite"]:.2f}  {v["medNon"]:7.4g} '
              f'{v["medNstar"]:7.4g} {v["ratio"]:.3f} | {v["medAucEnd"]:.3f} {v["medPred"]:.3f} {v["predErr"]:.3f} | '
              f'{v["medAucWEnd"]:.3f} {v["medCostW10"]:+.3f}  | {v["lock"]:.2f}  {v["pos"]:.2f}  {v["beta_te"]:+.2f}')
        if p["block"] == "A" and p["te"] == "sym":
            for k in ("R1", "R2", "R4a", "R4b"):
                tot[k] += 1
            if not (v["medPeak"] >= 0.80 and v["onFinite"] >= 0.80 and v["medNon"] <= 2e4):
                fails["R1"].append(tag(p))
            if not (math.isfinite(v["ratio"]) and 0.67 <= v["ratio"] <= 1.5):
                fails["R2"].append(tag(p))
            if v["medAucWEnd"] < 0.90:
                fails["R4a"].append(tag(p))
            if v["medCostW10"] < -0.02:
                fails["R4b"].append(tag(p))
        else:
            tot["R3"] += 1
            if v["predErr"] > 0.10:
                fails["R3"].append(tag(p))
    print()
    for a in sorted(load("C"), key=lambda r: r["params"]["dict"]):
        d, v = a["params"]["dict"], a["values"]
        print(f'C {d}: Ksub_l*={v["ksub_lstar"]:+.4f} slope={v["slope"]:+.3f} (pred {v["slope_pred"]:+.2f}) '
              + " ".join(f'n={n}: lock {v[f"lock_n{n}"]:.3f} pos {v[f"pos_n{n}"]:.3f}'
                         for n in (2000, 8000, 32000, 128000, 512000))
              + f' ks_H={v["ks_H"]:.3f}')
        tot["R5"] += 1
        if abs(v["slope_err"]) > 0.10:
            fails["R5"].append(d)
        if d == "oo4":
            tot["R6"] += 1
            if not (v["lock_n512000"] >= 0.95 and v["ks_H"] <= 0.08):
                fails["R6"].append(d)
        if d == "oe4":
            tot["R7"] += 1
            if abs(v["pos_n512000"] - 0.5) > 0.087:
                fails["R7"].append(d)
    print("\nVerdicts")
    for k in fails:
        f = fails[k]
        print(f'{k}: {"PASS" if not f else "FAIL"} ({tot[k] - len(f)}/{tot[k]})' + (f' — {"; ".join(f)}' if f else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
