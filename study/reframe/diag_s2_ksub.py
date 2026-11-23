"""Post-hoc check (not registered): the population quantities behind Assumption (B3) and Theorem 7
in the S2 crossover cells -- K^sub, its top coordinate K^sub_{l*}, and mu^te_L -- for the Gaussian
and the RflyMAD clean laws at a* = 8, training split (0.05, 0.05), symmetric test contamination.
Asked by the round-2 review: is RflyMAD oe near the excluded boundary K^sub_{l*} = 0?
Writes results/diag_s2_ksub.json.
"""
import json
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s2 as S
from spec_s2 import TE

OUT = Path(__file__).parent / "results" / "diag_s2_ksub.json"
SPLIT = (0.05, 0.05)


def main() -> None:
    cont = S.Contaminant("pareto")
    gauss = (S.Gauss(*H.G_PLUS), S.Gauss(*H.G_MINUS))
    P = S.rfly_pools(H.seed_of({"rfly_split": 1}))
    out = {}
    for src, tr, te in (("gauss", gauss, gauss), ("rfly", P["train"], P["test"])):
        for d in ("oo", "ee", "oe", "eo"):
            cols = H.columns(d, 8.0)
            ks = S.ksub(cols, SPLIT, tr, cont)
            dL = S.test_mean_diff_L(cols, TE["sym"], te, cont)
            out[f"{src}_{d}"] = {"Ksub": [float(x) for x in ks], "Ksub_lstar": float(ks[-1]),
                                 "rel_lstar": float(abs(ks[-1]) / np.max(np.abs(ks))),
                                 "mu_te_L": float(ks @ dL)}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, round(v["Ksub_lstar"], 4), round(v["rel_lstar"], 3), round(v["mu_te_L"], 4))


if __name__ == "__main__":
    main()
