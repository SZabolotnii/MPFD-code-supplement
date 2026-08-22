#!/usr/bin/env python3
"""LC1: half-line decomposition identities (L1) and the ceiling inequality."""
from __future__ import annotations

import numpy as np

from _common import SEED, write_result
from mcp import basis as B, laws, panel
from mcp import projection as P
from mcp.moments import OneSidedMoments, SampleOneSidedMoments

A, Bexp, SHIFT, NMC = 0.8, 1.4, 0.3, 4_000_000
rng = np.random.default_rng(SEED)
out = {"a": A, "b": Bexp, "shift": SHIFT, "laws": {}}
worst = {"identity": 0.0, "cross": 0.0, "mixed": 0.0, "schur": 0.0}

for law in (laws.laplace(), laws.student_t(6), panel.A2L):
    om, on = OneSidedMoments(law), OneSidedMoments(laws.affine(law, shift=SHIFT))
    H = B.halfline([A], [Bexp])
    cm, cn = P.population_moments(H, om), P.population_moments(H, on)
    j_solver = P.deflection(cm, cn).J
    j_formula = P.j_halfline_s3(om, on, A, Bexp)
    cross = max(abs(cm.C[1, 2] + om.nu(A, 1) * om.nu(Bexp, -1)), abs(cn.C[1, 2] + on.nu(A, 1) * on.nu(Bexp, -1)))
    jm = P.deflection(P.population_moments(B.mixed([A]), om), P.population_moments(B.mixed([A]), on)).J
    jh = P.deflection(P.population_moments(B.halfline([A], [A]), om), P.population_moments(B.halfline([A], [A]), on)).J
    M2 = B.mixed([2.0])
    cm2, cn2 = P.population_moments(M2, om), P.population_moments(M2, on)
    j_head, yt, s = P.schur_gain(cm2, cn2)
    j_cl = P.deflection(P.population_moments(B.classical(2), om), P.population_moments(B.classical(2), on)).J
    schur_err = max(abs(j_head - j_cl), abs(P.deflection(cm2, cn2).J - j_head - yt ** 2 / s))
    # (v) MC moments
    x = law.rvs(NMC, rng)
    smp = SampleOneSidedMoments(x)
    mc = {}
    for q in (0.8, 1.4, 1.6, 2.0):
        for side in (1, -1):
            pop, est = om.nu(q, side), smp.nu(q, side)
            se = np.std(np.where(side * x > 0, np.abs(x) ** q, 0.0)) / np.sqrt(NMC)
            mc[f"nu{'+' if side > 0 else '-'}({q})"] = {"pop": pop, "mc": est, "rel_err": abs(est - pop) / pop, "se": se,
                                                        "pass": abs(est - pop) / pop < 5e-3 and abs(est - pop) < 3 * se}
    out["laws"][law.name] = {"J_solver": j_solver, "J_formula": j_formula, "cross_block_err": cross,
                             "J_mixed": jm, "J_halfline_equal": jh, "schur": {"J_head": j_head, "Ytilde": yt, "S": s,
                             "J_classical2": j_cl, "J_mixed2": P.deflection(cm2, cn2).J}, "mc": mc}
    worst["identity"] = max(worst["identity"], abs(j_solver - j_formula))
    worst["cross"] = max(worst["cross"], cross)
    worst["mixed"] = max(worst["mixed"], abs(jm - jh))
    worst["schur"] = max(worst["schur"], schur_err)

# (vi) Gaussian scale pair: monotone in s and below the ceiling
lm, ln = laws.gaussian(), laws.affine(laws.gaussian(), scale=1.5)
om, on = OneSidedMoments(lm), OneSidedMoments(ln)
delta, jinf = P.j_infinity(lm, ln)
js = [P.deflection(P.population_moments(B.classical(s), om), P.population_moments(B.classical(s), on)).J for s in (1, 2, 3, 4)]
out["gaussian_ceiling"] = {"Delta": delta, "J_inf": jinf, "J_by_s": js,
                           "monotone": all(js[i] <= js[i + 1] + 1e-12 for i in range(3)), "below": js[-1] <= jinf}
out["identity_max_abs_err"] = worst["identity"]; out["cross_block_max_err"] = worst["cross"]
out["mixed_vs_halfline_err"] = worst["mixed"]; out["schur_check"] = worst["schur"]
out["mc_all_pass"] = all(v["pass"] for l in out["laws"].values() for v in l["mc"].values())
out["thresholds"] = {"algebra": 1e-10, "mc_rel": 5e-3}
out["PASS"] = max(worst.values()) < 1e-10 and out["mc_all_pass"] and out["gaussian_ceiling"]["monotone"] and out["gaussian_ceiling"]["below"]
print({k: out[k] for k in ("identity_max_abs_err", "cross_block_max_err", "mixed_vs_halfline_err", "schur_check", "mc_all_pass", "PASS")})
print("written", write_result("l01", out))
