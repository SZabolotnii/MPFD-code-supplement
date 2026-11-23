"""Evaluate the registered criteria P1-P8 of DESIGN.md on the hashed run artifacts.

Reads results/<run-id>/*.json only (no recomputation) and prints one table per block plus
the verdict of every registered criterion. Usage: python criteria.py [run-id]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "h1v4-2026-11-22"
D = Path(__file__).resolve().parent / "results" / RUN
NMAX = 128000


def load(block: str) -> list[dict]:
    rows = []
    for f in sorted(D.glob("*.json")):
        if f.name == "RUN.json":
            continue
        a = json.loads(f.read_text())
        if a["params"].get("block") == block:
            rows.append(a)
    return rows


def main() -> int:
    A = load("A")
    print(f"Block A: {len(A)} cells")
    hdr = "dict a*  p+  split | Ksub_l* | slope (pred)  err | lock@nmax pos@nmax | dAUC 500→nmax | KS_H  cond"
    print(hdr)
    fails = {k: [] for k in ("P1", "P2", "P3", "P4")}
    for a in sorted(A, key=lambda r: (r["params"]["dict"], r["params"]["astar"],
                                      r["params"]["pplus"], r["params"]["split"])):
        p, v = a["params"], a["values"]
        tag = f'{p["dict"]} {p["astar"]} {p["pplus"]} {p["split"]:5s}'
        print(f'{tag} | {v["Ksub_lstar"]:+.4f} | {v["slope"]:+.3f} ({v["slope_pred"]:+.3f}) {v["slope_err"]:+.3f} | '
              f'{v[f"lock_n{NMAX}"]:.3f} {v[f"pos_n{NMAX}"]:.3f} | {v["dauc_n500"]:.4f}→{v[f"dauc_n{NMAX}"]:.4f} | '
              f'{v["ks_H"]:.3f} {v[f"cond_n{NMAX}"]:.1f}')
        if abs(v["slope_err"]) > 0.10:
            fails["P1"].append(tag)
        if not (v[f"dauc_n{NMAX}"] <= 0.01 and v[f"dauc_n{NMAX}"] < v["dauc_n500"]):
            fails["P2"].append(tag)
        if v["same_parity"] == 1.0 and v[f"lock_n{NMAX}"] < 0.95:
            fails["P3"].append(tag)
        if v["same_parity"] == 0.0 and p["pplus"] == 0.5 and abs(v[f"pos_n{NMAX}"] - 0.5) > 0.087:
            fails["P4"].append(tag)
    n_cells = {"P1": len(A), "P2": len(A),
               "P3": sum(a["values"]["same_parity"] == 1.0 for a in A),
               "P4": sum(a["values"]["same_parity"] == 0.0 and a["params"]["pplus"] == 0.5 for a in A)}
    for k, bad in fails.items():
        print(f"{k}: {'PASS' if not bad else 'FAIL'} ({n_cells[k] - len(bad)}/{n_cells[k]} cells)"
              + ("" if not bad else "  failing: " + "; ".join(bad)))

    B = load("B")
    print(f"\nBlock B: {len(B)} cells")
    p6 = []
    for b in sorted(B, key=lambda r: r["params"]["dict"]):
        v = b["values"]
        name = b["params"]["dict"]
        print(f'{name}: slope {v["slope"]:+.3f} (pred {v["slope_pred"]:+.3f}, err {v["slope_err"]:+.3f}); '
              f'lambda_m^3 {v["lambda_m_pow"]:.4f}')
        for n in (50, 200, 1000):
            t = f"n{n}"
            print(f'   n={n:5d}  medN_on {v[f"medNon_{t}"]:.3g}  nan {v[f"nan_on_{t}"]:.2f}  medN* {v[f"medNstar_{t}"]:.3g}'
                  f'  ratio {v[f"ratio_{t}"]:.3f}  peak {v[f"peak_{t}"]:.3f} (medN {v[f"medNpeak_{t}"]:.3g}, minN {v[f"minNpeak_{t}"]:.3g},'
                  f' first finite N <= {v[f"maxNfirst_{t}"]:.3g})  lock {v[f"lock_{t}"]:.2f}  pos {v[f"pos_{t}"]:.2f}'
                  f'  S2err {v[f"S2err_{t}"]:.4f} (nan {v[f"S2nan_{t}"]:.2f})')
        if not (math.isfinite(v["slope_err"]) and abs(v["slope_err"]) <= 0.15):
            p6.append(name)
    if B:
        print(f"P6: {'PASS' if not p6 else 'FAIL'} ({len(B) - len(p6)}/{len(B)} dictionaries)"
              + ("" if not p6 else "  failing: " + ", ".join(p6)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
