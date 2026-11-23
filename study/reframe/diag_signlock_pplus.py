"""Post-hoc check (not registered): the general-p_+ limit of Theorem 5(ii) for opposite parity,
P{K_t > 0} -> rho(alpha_c/(a*+a_L), -sgn(Ksub_l*)(2p_+ - 1)), against the stored run-1 sign
frequencies (pos_n500 ... pos_n128000) of the opposite-parity Block A cells. The prediction needs
only the population vector Ksub (h1v4.ksub, closed-form moments); it is also checked against the
H sampler, P(H > 0) with the full tail variance. Added with the stage-5 amendment of 2026-11-23.
Writes results/diag_signlock_pplus.json.
"""
import json
from pathlib import Path

import numpy as np

import h1v4 as H
from spec_h1v4 import SPLITS

RUN = Path(__file__).parent / "results" / "h1v4-2026-11-22"
OUT = Path(__file__).parent / "results" / "diag_signlock_pplus.json"


def main() -> None:
    out = {}
    for f in sorted(RUN.glob("*block=A*.json")):
        d = json.loads(f.read_text())
        p, v = d["params"], d["values"]
        if v["same_parity"] == 1.0:
            continue
        cols = H.columns(p["dict"], p["astar"])
        ks = H.ksub(cols, SPLITS[p["split"]], p["pplus"])
        g = H.ALPHA / (p["astar"] + cols[-2][1])
        beta = 2 * p["pplus"] - 1
        pred = H.rho(g, -float(np.sign(ks[-1])) * beta)
        hkey = f'{p["dict"]}_a{p["astar"]}_p{p["pplus"]}'
        h = H.h_sample(cols, p["pplus"], 20000,
                       np.random.default_rng(H.seed_of({"H": p["dict"], "astar": p["astar"], "p": p["pplus"]})),
                       campbell=True)
        pH = float(np.mean(h > 0))
        out[f.name] = {"dict": p["dict"], "astar": p["astar"], "pplus": p["pplus"], "split": p["split"],
                       "ksub_lstar": float(ks[-1]), "gamma_prime": g, "pred": pred,
                       "pred_sampler": pH if ks[-1] < 0 else 1 - pH, "h_key": hkey,
                       **{k: v[k] for k in v if k.startswith("pos_n")}}
    OUT.write_text(json.dumps(out, indent=1))
    for v in out.values():
        print(v["dict"], v["astar"], v["pplus"], v["split"], f'pred={v["pred"]:.3f}',
              f'sampler={v["pred_sampler"]:.3f}', f'pos128k={v["pos_n128000"]:.3f}')


if __name__ == "__main__":
    main()
