"""Design-time probe for S3-A (no window AUC is computed): for each candidate cell, the median of
N*_n(K_hat) over 40 training fits, the population K^sub_{l*} and mu^te_L, and the predicted onset
lambda_m^{alpha/(a*-alpha)} * median N*. Cells whose predicted onset falls inside [50, 10^4] are the
ones DESIGN_S3.md may register. Writes results/design_probe_s3.json.
"""
import json
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T

OUT = Path(__file__).parent / "results" / "design_probe_s3.json"
EPS_TR = (0.05, 0.05)
TE = {"sym": (0.05, 0.05), "one": (0.10, 0.0)}


def main() -> None:
    out = {}
    for alpha in (1.2, 1.5):
        cont = T.Pareto(alpha)
        for astar in (3.0, 4.0):
            gam = alpha / astar
            lam_pow = H.lambda_mid(gam, 0.75) ** (alpha / (astar - alpha))
            for name in ("oo", "oe", "ee", "eo"):
                cols = T.columns(name, astar, alpha)
                ks = S.ksub(cols, EPS_TR, T.G_CLEAN, cont)
                for n in (200, 1000):
                    rng = np.random.default_rng(H.seed_of({"probe_s3": name, "a": astar, "al": alpha, "n": n}))
                    fits = []
                    for _ in range(40):
                        xp = S.draw(rng, n, EPS_TR[0], T.G_CLEAN[0], cont)
                        xm = S.draw(rng, n, EPS_TR[1], T.G_CLEAN[1], cont)
                        fits.append(H.fit(cols, xp, xm)[0])
                    for te, eps_te in TE.items():
                        if te == "one" and cols[-1][0] != "e":
                            continue
                        ns = np.array([T.nstar(cols, K, eps_te, T.G_CLEAN, cont) for K in fits])
                        mu = float(ks @ S.test_mean_diff_L(cols, eps_te, T.G_CLEAN, cont))
                        key = f"{name}_a{astar}_al{alpha}_n{n}_{te}"
                        out[key] = {"cols": cols, "ksub_lstar": float(ks[-1]), "mu_te_L": mu,
                                    "medNstar": float(np.median(ns)), "lam_pow": lam_pow,
                                    "pred_onset": float(lam_pow * np.median(ns)),
                                    "beta_te": S.beta_te(cols, eps_te, cont)}
                        print(key, f'Ksub*={ks[-1]:+.3f} mu={mu:+.3f} medN*={np.median(ns):.3g} '
                                   f'pred_on={lam_pow * np.median(ns):.3g}')
    # S3-B candidates: alpha_c = 3, a* = 8, n = 200, even tops, interior and reverse test splits
    cont = T.Pareto(3.0)
    lam_pow = H.lambda_mid(3.0 / 8.0, 0.75) ** (3.0 / 5.0)
    for name in ("oe", "ee"):
        cols = T.columns(name, 8.0, 3.0)
        ks = S.ksub(cols, EPS_TR, T.G_CLEAN, cont)
        rng = np.random.default_rng(H.seed_of({"probe_s3b": name}))
        fits = [H.fit(cols, S.draw(rng, 200, EPS_TR[0], T.G_CLEAN[0], cont),
                      S.draw(rng, 200, EPS_TR[1], T.G_CLEAN[1], cont))[0] for _ in range(40)]
        for te, eps_te in {"in+": (0.3, 0.1), "in-": (0.1, 0.3), "rev10": (0.0, 0.1),
                           "rev40": (0.0, 0.4)}.items():
            ns = np.array([T.nstar(cols, K, eps_te, T.G_CLEAN, cont) for K in fits])
            mu = float(ks @ S.test_mean_diff_L(cols, eps_te, T.G_CLEAN, cont))
            key = f"B_{name}_{te}"
            out[key] = {"cols": cols, "ksub_lstar": float(ks[-1]), "mu_te_L": mu,
                        "medNstar": float(np.median(ns)), "lam_pow": lam_pow,
                        "pred_onset": float(lam_pow * np.median(ns)),
                        "beta_te": S.beta_te(cols, eps_te, cont),
                        "share_mu_nonpos": float(np.mean(ns == 1.0))}
            print(key, f'Ksub*={ks[-1]:+.3f} mu={mu:+.3f} beta={out[key]["beta_te"]:+.2f} '
                       f'medN*={np.median(ns):.3g} pred_on={lam_pow * np.median(ns):.3g} '
                       f'mu<=0 share={out[key]["share_mu_nonpos"]:.2f}')
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
