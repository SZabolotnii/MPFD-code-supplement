"""Evaluate the registered criteria Q1-Q5 of DESIGN_S2.md on the hashed S2 artifacts.

Reads results/<run-id>/*.json only (no recomputation). Usage: python criteria_s2.py [run-id]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "h1v4-s2-2026-11-22"
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
    return f'{p["src"]:5s} {p["cont"]:6s} {p["dict"]} n={p["n"]:<4d} {p["te"]}'


def main() -> int:
    X = sorted(load("X"), key=lambda r: (r["params"]["src"], r["params"]["cont"], r["params"]["te"],
                                         r["params"]["dict"], r["params"]["n"]))
    print(f"Crossover cells: {len(X)}")
    print("cell                               | fail | peak  onFin medNon  medN*  ratio | end   pred-err | endW  endS  costW10 | lock  pos   cond")
    fails = {k: [] for k in ("Q1", "Q2", "Q3", "Q5a", "Q5b")}
    for a in X:
        p, v = a["params"], a["values"]
        print(f'{tag(p):34s} | {v["failed"]:.0f}    | {v["medPeak"]:.3f} {v["onFinite"]:.2f}  {v["medNon"]:7.4g} {v["medNstar"]:7.4g} '
              f'{v["ratio"]:.3f} | {v["medAucEnd"]:.3f} {v["predErr"]:.3f}    | {v["medAucWEnd"]:.3f} {v["medAucSEnd"]:.3f} '
              f'{v["medCostW10"]:+.3f}  | {v["lock"]:.2f}  {v["pos"]:.2f}  {v["medCond"]:.3g}')
        if p["te"] == "sym":
            ok1 = v["medPeak"] >= 0.80 and v["onFinite"] >= 0.80 and v["medNon"] <= 2e4
            if not ok1:
                fails["Q1"].append(tag(p))
            if p["src"] != "rfly" and not (math.isfinite(v["ratio"]) and 0.67 <= v["ratio"] <= 1.5):
                fails["Q2"].append(tag(p))
            if v["medAucWEnd"] < 0.90:
                fails["Q5a"].append(tag(p))
            if v["medCostW10"] < -0.02:
                fails["Q5b"].append(tag(p))
        else:
            if v["predErr"] > 0.10:
                fails["Q3"].append(tag(p))
    T = load("T")
    print(f"\nt3 training cells: {len(T)}")
    q4 = []
    for a in sorted(T, key=lambda r: r["params"]["dict"]):
        p, v = a["params"], a["values"]
        print(f'{p["dict"]}: Ksub_l*={v["Ksub_lstar"]:+.4f} ' + " ".join(
            f'n={n}: lock {v[f"lock_n{n}"]:.3f} pos {v[f"pos_n{n}"]:.3f} dAUC {v[f"dauc_n{n}"]:.4f}'
            for n in (2000, 32000, 128000)))
        if p["dict"] == "ee" and v["lock_n128000"] < 0.95:
            q4.append("ee lock")
        if v["dauc_n128000"] > 0.01:
            q4.append(f'{p["dict"]} dAUC')
    fails["Q4"] = q4
    nsym = sum(1 for a in X if a["params"]["te"] == "sym")
    none = sum(1 for a in X if a["params"]["te"] == "one")
    nq2 = sum(1 for a in X if a["params"]["te"] == "sym" and a["params"]["src"] != "rfly")
    tot = {"Q1": nsym, "Q2": nq2, "Q3": none, "Q4": 3, "Q5a": nsym, "Q5b": nsym}
    print("\nVerdicts")
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5a", "Q5b"):
        f = fails[k]
        print(f'{k}: {"PASS" if not f else "FAIL"} ({tot[k] - len(f)}/{tot[k]})' + (f' — {"; ".join(f)}' if f else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
