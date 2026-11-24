"""Post-hoc diagnostic (after run 4, not registered): does the clean drift of the top column explain
the onset ratios of the symmetric block-A cells of run 4 ((x, x^3) and (x, |x|^3), alpha_c in
{2.2, 2.5})?

Same construction as diag_b_clean_drift.py (run 1). N*_n(K_hat) uses the admissible drift
mu_hat = K_L' (E+ phi_L - E- phi_L) only. At finite n the window score also carries the clean drift
of the top column, K_t * {(1 - eps+) E_{G+} phi_t - (1 - eps-) E_{G-} phi_t}; under symmetric test
contamination the contaminant parts of phi_t enter only through the stable term T_N. Replacing mu_hat
by mu_eff = mu_hat + that term gives N*_eff.

Refits K_hat with the run's own seeds (training draws only, no windows), checks that the stored
per-fit N* are reproduced, and compares the per-fit onset ratio N_on / (lambda_m^{alpha/(a*-alpha)} N*)
with N_on / (lambda_m^{alpha/(a*-alpha)} N*_eff). Writes results/diag_s4_clean_drift.json.
"""
import json
import math
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T
import h1v4_s4 as F
import spec_s4

RUN = Path(__file__).parent / "results" / "h1v4-s4-2026-11-24"
OUT = Path(__file__).parent / "results" / "diag_s4_clean_drift.json"
EPS_TR = spec_s4.EPS_TR


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, (0.25, 0.5, 0.75))] if x.size else [math.nan] * 3


def cell(path: Path) -> dict:
    d = json.loads(path.read_text())
    p, v = d["params"], d["values"]
    cols = [tuple(c) for c in F.DICTS[p["dict"]]]
    alpha, astar, n = p["alpha"], 3.0, p["n"]
    eps_te = spec_s4.TE_A[p["te"]]
    cont = T.Pareto(alpha)
    tr, te = F.cleans_of("gauss")
    par, at = cols[-1]
    dt = (1 - eps_te[0]) * te[0].moment(par, at) - (1 - eps_te[1]) * te[1].moment(par, at)
    dL = S.test_mean_diff_L(cols, eps_te, te, cont)
    e = cont.c0 * sum(eps_te)
    lam_pow = H.lambda_mid(alpha / astar, 0.75) ** (alpha / (astar - alpha))
    ns, ne = [], []
    for s in F._seeds(H.seed_of(p)):
        rng = np.random.default_rng(s)
        xp = S.draw(rng, n, EPS_TR[0], tr[0], cont)
        xm = S.draw(rng, n, EPS_TR[1], tr[1], cont)
        K, _ = H.fit(cols, xp, xm)
        ns.append(T.nstar(cols, K, eps_te, te, cont))
        mu_eff = float(K[:-1] @ dL) + K[-1] * dt
        ne.append((mu_eff / abs(K[-1])) ** (alpha / (at - alpha)) * e ** (-at / (at - alpha))
                  if mu_eff > 0 and K[-1] != 0 else math.nan)
    ns, ne = np.array(ns), np.array(ne)
    assert np.allclose(ns, v["fit_Nstar"], rtol=1e-9), path.name
    on = np.array(v["fit_Non"])
    fin = np.isfinite(on)
    ok = fin & np.isfinite(ne)
    return {"file": path.name, "clean_top_diff": dt, "n_onset": int(fin.sum()),
            "n_onset_eff_defined": int(ok.sum()),
            "ratio_q": q(on[fin] / (lam_pow * ns[fin])),
            "ratio_eff_q": q(on[ok] / (lam_pow * ne[ok])),
            "share_mu_eff_nonpos": float(np.mean(~np.isfinite(ne))),
            "pos": float(np.mean(np.array(v["fit_Kt"]) > 0))}


if __name__ == "__main__":
    res = {}
    for f in sorted(RUN.glob("*block=A4*.json")):
        p = json.loads(f.read_text())["params"]
        if p["dict"] == "poly3" or not p["te"].startswith("sym"):
            continue
        key = f'{p["dict"]}_al{p["alpha"]}_n{p["n"]}_{p["te"]}'
        res[key] = cell(f)
        r = res[key]
        print(f'{key:24s} dt={r["clean_top_diff"]:+.3f} pos={r["pos"]:.2f} onsets={r["n_onset"]} '
              f'ratio q25/50/75={["%.2f" % x for x in r["ratio_q"]]} eff={["%.2f" % x for x in r["ratio_eff_q"]]} '
              f'(n={r["n_onset_eff_defined"]}, mu_eff<=0 share {r["share_mu_eff_nonpos"]:.2f})')
    OUT.write_text(json.dumps(res, indent=1))
