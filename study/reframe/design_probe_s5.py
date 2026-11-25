"""Design-time probe for S5 (no window AUC of block G is computed here).

Block F: kappa_0 by the registered rule and the window lengths N_n it gives; one fitted vector per
dictionary at the largest n through the exact AUC, with its conditional limit, to see that the CF
engine works at N_n.
Block G: for each candidate cell of (x, |x|^3), 40 training fits give P{K_t > 0}, the share of fits
with sgn(K_t) beta_te = -1 (the collapsing fits), their median N*_n, the share with mu_hat <= 0 and
the clean top drift. DESIGN_S5.md registers the cells from this output before any window exists.
Writes results/design_probe_s5.json.
"""
import json
import math
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
import h1v4_s3 as T
import h1v4_s5 as G

OUT = Path(__file__).parent / "results" / "design_probe_s5.json"
EPS_TR = (0.05, 0.05)


def probe_g(alpha, n, src, out):
    cols = [tuple(c) for c in G.G_COLS]
    cont = T.Pareto(alpha)
    tr, te = G.g_cleans(src)
    rng = np.random.default_rng(H.seed_of({"probe_s5": src, "alpha": alpha, "n": n}))
    fits = [H.fit(cols, S.draw(rng, n, EPS_TR[0], tr[0], cont), S.draw(rng, n, EPS_TR[1], tr[1], cont))[0]
            for _ in range(40)]
    pos = float(np.mean([K[-1] > 0 for K in fits]))
    par, at = cols[-1]
    for side in ("fwd", "rev"):
        for e in (0.1, 0.2, 0.4):
            eps_te = (e, 0.0) if side == "fwd" else (0.0, e)
            beta = S.beta_te(cols, eps_te, cont)
            ns = np.array([T.nstar(cols, K, eps_te, te, cont) for K in fits])
            coll = np.array([np.sign(K[-1]) * beta < 0 for K in fits])
            dt = (1 - eps_te[0]) * te[0].moment(par, at) - (1 - eps_te[1]) * te[1].moment(par, at)
            dL = S.test_mean_diff_L(cols, eps_te, te, cont)
            ee = cont.c0 * sum(eps_te)
            ne = []
            for K in fits:
                mu = float(K[:-1] @ dL) + K[-1] * dt
                ne.append((mu / abs(K[-1])) ** (alpha / (at - alpha)) * ee ** (-at / (at - alpha)) if mu > 0 else 1.0)
            ne = np.array(ne)
            k = f"G_{src}_al{alpha}_n{n}_{side}{int(round(e * 100)):02d}"
            out[k] = {"pos": pos, "beta_te": beta, "share_collapse": float(coll.mean()),
                      "coll_medNstar": float(np.median(ns[coll])) if coll.any() else math.nan,
                      "coll_medNstar_eff": float(np.median(ne[coll])) if coll.any() else math.nan,
                      "share_mu_nonpos": float(np.mean(ns == 1.0)), "clean_top_diff": float(dt)}
            print(f"{k:30s} pos={pos:.2f} beta={beta:+.0f} coll={coll.mean():.2f} "
                  f"medN*_coll={out[k]['coll_medNstar']:.3g} medN*eff_coll={out[k]['coll_medNstar_eff']:.3g} "
                  f"mu<=0={out[k]['share_mu_nonpos']:.2f} dt={dt:+.3f}", flush=True)


def probe_f(out):
    for name in G.F_DICTS:
        P = G.f_population(name)
        k0 = G.kappa0_rule(name)
        Ns = {n: G.window_length(k0, n, P["astar"], P["aL"]) for n in G.F_NS[-3:]}
        K = G.f_fits(name, G.F_NS[-1])[0]
        r = G.f_auc_job((name, K.tolist(), Ns[G.F_NS[-1]]))
        out[f"F_{name}"] = {"kappa0": k0, "N_n": {str(n): v for n, v in Ns.items()},
                            "ksub_lstar": P["ksub_lstar"], "mu_te_L": P["mu_te_L"], "beta_te": P["beta_te"],
                            "lambda_m": P["lambda_m"], "probe_auc_first_fit_nmax": r,
                            "probe_Kt_first_fit": float(K[-1])}
        print(f"F_{name}: kappa0={k0:.4g} N_n={Ns} first-fit AUC={r} Kt={K[-1]:.3g}", flush=True)


def main() -> None:
    out = {}
    probe_f(out)
    for src in ("gauss", "mirror"):
        for alpha in (2.2, 2.5):
            for n in (200, 1000):
                probe_g(alpha, n, src, out)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
