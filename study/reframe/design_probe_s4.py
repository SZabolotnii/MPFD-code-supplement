"""Design-time probe for S4 blocks A, B, C, E (no window AUC is computed): for each candidate cell,
40 training fits give the median N*_n(K_hat), the lock share, the population K^sub_{l*} and
mu^te_L, and the predicted onset lambda_m^{alpha/(a*-alpha)} * median N*. DESIGN_S4.md registers
the cells from this output before any window exists. Writes results/design_probe_s4.json.
"""
import json
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T
import h1v4_s4 as F

OUT = Path(__file__).parent / "results" / "design_probe_s4.json"
EPS_TR = (0.05, 0.05)


def probe(key, cols, astar, alpha, n, splits, src, out):
    cont = T.Pareto(alpha)
    tr, te = F.cleans_of(src)
    ks = S.ksub(cols, EPS_TR, tr, cont)
    lam_pow = H.lambda_mid(alpha / astar, 0.75) ** (alpha / (astar - alpha))
    rng = np.random.default_rng(H.seed_of({"probe_s4": key, "n": n}))
    fits = [H.fit(cols, S.draw(rng, n, EPS_TR[0], tr[0], cont), S.draw(rng, n, EPS_TR[1], tr[1], cont))[0]
            for _ in range(40)]
    lock = float(np.mean([np.sign(K[-1]) == -np.sign(ks[-1]) for K in fits]))
    for te_name, eps_te in splits.items():
        ns = np.array([T.nstar(cols, K, eps_te, te, cont) for K in fits])
        mu = float(ks @ S.test_mean_diff_L(cols, eps_te, te, cont))
        k = f"{key}_n{n}_{te_name}"
        out[k] = {"cols": [list(c) for c in cols], "ksub_lstar": float(ks[-1]), "mu_te_L": mu,
                  "lock": lock, "medNstar": float(np.median(ns)), "lam_pow": lam_pow,
                  "pred_onset": float(lam_pow * np.median(ns)),
                  "beta_te": S.beta_te(cols, eps_te, cont), "share_mu_nonpos": float(np.mean(ns == 1.0))}
        print(f"{k:34s} Ksub*={ks[-1]:+.3f} mu={mu:+.4f} beta={out[k]['beta_te']:+.2f} lock={lock:.2f} "
              f"medN*={np.median(ns):.3g} pred_on={lam_pow * np.median(ns):.3g} "
              f"mu<=0={out[k]['share_mu_nonpos']:.2f}", flush=True)


def main() -> None:
    out = {}
    # A: (x, x^3), (x, |x|^3) at alpha_c in (2, 3)
    for alpha in (2.2, 2.5):
        for d in ("oo3", "oe3"):
            cols = list(F.DICTS[d])
            splits = {"sym05": (0.05, 0.05), "sym20": (0.2, 0.2)}
            if d == "oe3":
                splits |= {"one10": (0.1, 0.0), "one40": (0.4, 0.0)}
            for n in (50, 200, 1000):
                probe(f"A_{d}_al{alpha}", cols, 3.0, alpha, n, splits, "gauss", out)
    # B: locked interior beta
    cols = T.columns("ee", 4.0, 1.2)
    probe("B_ee_a4_al1.2", cols, 4.0, 1.2, 1000, {"in+": (0.075, 0.025), "in-": (0.025, 0.075)}, "gauss", out)
    # C: RflyMAD pools in the run-3 regime
    for alpha in (1.2, 1.5):
        for astar in (3.0, 4.0):
            for d in ("oe", "ee"):
                cols = T.columns(d, astar, alpha)
                for n in (200, 1000):
                    probe(f"C_{d}_a{astar}_al{alpha}", cols, astar, alpha, n,
                          {"sym": (0.05, 0.05), "one": (0.10, 0.0)}, "rfly", out)
    # E: scale-only clean laws, forward and reverse one-sided splits
    cols = T.columns("ee", 8.0, 3.0)
    probe("E_ee_a8_al3", cols, 8.0, 3.0, 200,
          {"fwd10": (0.1, 0.0), "rev10": (0.0, 0.1), "fwd40": (0.4, 0.0), "rev40": (0.0, 0.4)}, "scale", out)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
