"""Post-hoc diagnostic (after the S2 run, not registered): why the four Q1-failing cells lack an onset.

Recomputes, with the run's own seeds, the per-replication peak, N_on and N*_n(K_hat) of the cells
that failed Q1 on its "onset finite in >= 80 %" clause, and classifies each missing onset:
  low_peak   -- the area under the curve never exceeds 3/4 on the grid (collapse before or at
                the first window lengths: left-censored);
  no_drop    -- it peaks above 3/4 but does not fall back to 3/4 by N_max (right-censored).
Writes results/diag_s2_onset_censoring.json.  Run: python diag_s2_onset_censoring.py
"""
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
from spec_s2 import TE

RUN = Path(__file__).parent / "results" / "h1v4-s2-2026-11-22"
OUT = Path(__file__).parent / "results" / "diag_s2_onset_censoring.json"
CELLS = ("dict=oo_n=50_src=gauss_te=sym", "dict=ee_n=50_src=gauss_te=sym",
         "dict=eo_n=50_src=gauss_te=sym", "dict=oe_n=1000_src=rfly_te=sym")


def run_cell(tag: str) -> dict:
    f = next(p for p in RUN.glob("astar=8.0_block=X_cont=pareto_*.json") if tag in p.name)
    d = json.loads(f.read_text())
    prm = d["params"]
    seed = H.seed_of(prm)
    jobs = [(prm["dict"], prm["astar"], prm["n"], TE[prm["te"]], prm["cont"], prm["src"],
             int(c.generate_state(1)[0])) for c in np.random.SeedSequence(seed).spawn(S.R_X)]
    with mp.get_context("spawn").Pool(8) as pool:
        reps = [r for r in pool.map(S.crossover_rep, jobs) if r["failed"] == 0.0]
    on = np.array([r["N_on"] for r in reps])
    peak = np.array([r["peak"] for r in reps])
    ns = np.array([r["Nstar"] for r in reps])
    fin = np.isfinite(on)
    low = (~fin) & (peak <= 0.75)
    nod = (~fin) & (peak > 0.75)
    med = lambda x: float(np.median(x)) if x.size else float("nan")
    out = {"file": f.name, "n_reps": len(reps), "onFinite": float(fin.mean()),
           "frac_low_peak": float(low.mean()), "frac_no_drop": float(nod.mean()),
           "medNstar_all": med(ns), "medNstar_finite": med(ns[fin]),
           "medNstar_low_peak": med(ns[low]), "medNstar_no_drop": med(ns[nod]),
           "check_medNon": med(on[fin]), "run_medNon": d["values"]["medNon"],
           "N_first": float(S.N_GRID[0]), "N_max": float(S.N_GRID[-1])}
    assert math.isclose(out["check_medNon"], out["run_medNon"], rel_tol=1e-9), out
    return out


if __name__ == "__main__":
    res = {t: run_cell(t) for t in CELLS}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
