#!/usr/bin/env python3
"""LC4: equal-budget (3 columns) support split H vs parity split M vs PATP P for the LLR target (L5).

Fix round 1 (controller ruling): the registered symmetric-identity criterion compares the *joint*
optimum of H (over the full 2D grid) with M and is expected to hold to 1e-8; it does not, on LP2
(joint optimum is off-diagonal). The registered criterion and threshold are unchanged (PASS stays
tied to it). Two diagnostics are added to make the mechanism checkable from the JSON without
retuning anything: H_diag (H restricted to the diagonal a=b, which the identity actually governs)
and E (a free even pair {x, |x|^a, |x|^b} at the same 3-column budget, a<b)."""
from __future__ import annotations

import numpy as np

from _common import write_result
from mcp import basis as B, panel
from mcp import projection as P
from mcp.moments import OneSidedMoments

GRID = np.round(np.linspace(0.55, 1.95, 15), 3)
ALPHAS = np.linspace(0.05, 0.99, 40); ALPHAS = ALPHAS[np.abs(ALPHAS - 0.5) > 0.04]


def best(om, on, builder, grid_pts):
    res = []
    for args in grid_pts:
        cols = builder(*args)
        d = P.deflection(P.population_moments(cols, om), P.population_moments(cols, on))
        if d.cond <= P.COND_LIMIT:
            res.append((d.J, args, d.cond))
    J, args, cond = max(res, key=lambda t: t[0])
    return {"J": J, "args": [float(a) for a in args], "cond": cond, "n_excluded": len(grid_pts) - len(res)}


pairs = {**panel.lp1(), **panel.lp2(), **panel.lp3(), **panel.lp4()}
out = {"grid": GRID.tolist(), "alphas": ALPHAS.tolist(), "budget_columns": 3, "pairs": {}}
sym_err = 0.0
diag_err = 0.0
for name, (lm, ln) in pairs.items():
    om, on = OneSidedMoments(lm), OneSidedMoments(ln)
    delta, jinf = P.j_infinity(lm, ln)
    H = best(om, on, lambda a, b: B.halfline([a], [b]), [(a, b) for a in GRID for b in GRID])
    M = best(om, on, lambda a: B.mixed([a]), [(a,) for a in GRID])
    Pp = best(om, on, lambda al: B.patp(al, 3), [(al,) for al in ALPHAS])
    H_diag = best(om, on, lambda a: B.halfline([a], [a]), [(a,) for a in GRID])
    E = best(om, on, lambda a, b: [B.LINEAR, B.Column("abs", float(a)), B.Column("abs", float(b))],
             [(a, b) for i, a in enumerate(GRID) for b in GRID[i + 1:]])
    # The classical {x, x^2, x^3} control. Sec. 2 of the manuscript names it the control of the
    # budget experiment and until now only the contamination experiment carried it, so the
    # headline table had no baseline in it. It has no free exponent, so there is nothing to
    # search: one population evaluation, reported at whatever cond(F) it lands on. That cond is
    # kept rather than screened -- an ill-conditioned control is a result about the control, and
    # the COND_LIMIT screen exists to stop a grid search from *selecting* a singular optimum.
    # best() would screen it away instead of reporting it, so evaluate it directly.
    ccols = B.classical(3)
    cd = P.deflection(P.population_moments(ccols, om), P.population_moments(ccols, on))
    C = {"J": cd.J, "args": [], "cond": cd.cond, "n_excluded": 0}
    for r in (H, M, Pp, H_diag, E, C):
        r["frac"] = r["J"] / jinf
    diag_minus_M_rel = abs(H_diag["J"] - M["J"]) / M["J"]
    row = {"Delta": delta, "J_inf": jinf, "H": H, "M": M, "P": Pp, "H_diag": H_diag, "E": E, "C": C,
           "gain_H_over_M_pct": 100 * (H["J"] - M["J"]) / M["J"],
           "gain_H_over_E_pct": 100 * (H["J"] - E["J"]) / E["J"],
           "diag_minus_M_rel": diag_minus_M_rel,
           "symmetric_pair": lm.symmetric and ln.symmetric}
    if row["symmetric_pair"]:
        sym_err = max(sym_err, abs(H["J"] - M["J"]) / M["J"])
        diag_err = max(diag_err, diag_minus_M_rel)
    out["pairs"][name] = row
    print(f"{name:<42} J_inf={jinf:.4f}  H={H['frac']:.3f}{tuple(H['args'])}  M={M['frac']:.3f}{tuple(M['args'])}  "
          f"P={Pp['frac']:.3f}{tuple(np.round(Pp['args'],3))}  "
          f"H_diag={H_diag['frac']:.3f}{tuple(H_diag['args'])}  E={E['frac']:.3f}{tuple(E['args'])}  "
          f"gain_H/M={row['gain_H_over_M_pct']:+.2f}%  gain_H/E={row['gain_H_over_E_pct']:+.2f}%")
out["symmetric_identity_max_err"] = sym_err
out["diag_identity_max_err"] = diag_err
out["note"] = ("registered symmetric_identity criterion compares the joint optimum of H with M and "
               "fails on LP2; the diagonal-restricted identity is the diagnostic")
out["PASS"] = sym_err < 1e-8
print("PASS", out["PASS"], "| diag_identity_max_err", out["diag_identity_max_err"],
      "| written", write_result("l04", out))
