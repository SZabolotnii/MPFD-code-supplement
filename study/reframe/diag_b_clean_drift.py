"""Post-hoc diagnostic (after run 1, not registered): does the clean drift of the top column explain
the Block B onset ratios, in particular oe at n = 1000 (1.361)?

N*_n(K_hat) of the crossover theorem uses the admissible drift mu_hat = K_L' (E+ phi_L - E- phi_L)
only.  At finite n the window score also has the clean drift of the top column,
K_t * {(1 - eps+) E_{G+} phi_t - (1 - eps-) E_{G-} phi_t}, of order |K_t|; under the symmetric test
contamination of Block B the contaminant parts of phi_t cancel in the mean.  Replacing mu_hat by
mu_eff = mu_hat + that term gives N*_eff.  If the drift is the cause, the median of N*_eff / N*
over the replications should reproduce the stored onset ratio of each cell.

Refits K_hat with the run's own seeds (no characteristic-function evaluation) and checks that the
stored median N*_n is reproduced.  Writes results/diag_b_clean_drift.json.
"""
import json
import math
from pathlib import Path

import numpy as np

import h1v4 as H

RUN = Path(__file__).parent / "results" / "h1v4-2026-11-22"
OUT = Path(__file__).parent / "results" / "diag_b_clean_drift.json"
EPS = (0.05, 0.05)
PPLUS = 0.5


def clean_top_diff(cols) -> float:
    par, a = cols[-1]
    return ((1 - EPS[0]) * H._gauss_moment(par, a, *H.G_PLUS)
            - (1 - EPS[1]) * H._gauss_moment(par, a, *H.G_MINUS))


def cell(name: str) -> dict:
    f = next(RUN.glob(f"block=B_dict={name}__*.json"))
    d = json.loads(f.read_text())
    seed = H.seed_of(d["params"])
    cols = H.columns(name, 4.0)
    dL = H.test_mean_diff_L(cols, EPS, PPLUS)
    dt = clean_top_diff(cols)
    e, at = sum(EPS), cols[-1][1]
    out = {"file": f.name, "clean_top_diff": dt}
    ss = np.random.SeedSequence(seed)
    for n, child in zip(H.NS_B, ss.spawn(len(H.NS_B))):
        ns, ne, kt = [], [], []
        for c in child.spawn(H.R_B):
            rng = np.random.default_rng(int(c.generate_state(1)[0]))
            K, _ = H.fit(cols, H.draw(rng, n, EPS[0], H.G_PLUS, PPLUS), H.draw(rng, n, EPS[1], H.G_MINUS, PPLUS))
            mu = float(K[:-1] @ dL)
            mu_eff = mu + K[-1] * dt
            n0 = H.nstar(cols, K, EPS, PPLUS)
            n1 = ((mu_eff / abs(K[-1])) ** (H.ALPHA / (at - H.ALPHA)) * e ** (-at / (at - H.ALPHA))
                  if mu_eff > 0 and K[-1] != 0 else float("nan"))
            ns.append(n0), ne.append(n1), kt.append(K[-1])
        ns, ne = np.array(ns), np.array(ne)
        ok = np.isfinite(ne) & np.isfinite(ns)
        med0 = float(np.median(ns))
        assert math.isclose(med0, d["values"][f"medNstar_n{n}"], rel_tol=1e-9), (name, n, med0)
        out[f"n{n}"] = {"medNstar": med0, "med_ratio_eff": float(np.median(ne[ok] / ns[ok])),
                        "stored_onset_ratio": d["values"][f"ratio_n{n}"],
                        "pos": float(np.mean(np.array(kt) > 0)), "n_ok": int(ok.sum())}
    return out


if __name__ == "__main__":
    res = {d: cell(d) for d in ("oo", "ee", "oe", "eo")}
    OUT.write_text(json.dumps(res, indent=1))
    for d, r in res.items():
        print(d, {k: (round(v["med_ratio_eff"], 3), round(v["stored_onset_ratio"], 3)) for k, v in r.items() if k.startswith("n")})
