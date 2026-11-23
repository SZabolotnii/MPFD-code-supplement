"""Post-hoc check (not registered): how far the run-1 reference law of H moves when the tail term of
the opposite-parity sampler gets its full (Campbell) variance instead of the conditional 4 p+ p-.
Both laws are drawn from the same seed as in run 1, so they share every random number and differ
only in the scale of the normal tail term. By the triangle inequality
|ks(W, H_old) - ks(W, H_new)| <= ks(H_old, H_new), so the distance bounds the change of every stored
ks_H of that cell. Added during review of the manuscript. Writes results/diag_h_tail.json.
"""
import json
from pathlib import Path

import numpy as np

import h1v4 as H

OUT = Path(__file__).parent / "results" / "diag_h_tail.json"


def main() -> None:
    out = {}
    for d in ("oe", "eo"):
        for astar in (2.4, 4.0):
            for pplus in (0.5, 0.8):
                cols = H.columns(d, astar)
                seed = H.seed_of({"H": d, "astar": astar, "p": pplus})
                old = H.h_sample(cols, pplus, 20000, np.random.default_rng(seed))
                new = H.h_sample(cols, pplus, 20000, np.random.default_rng(seed), campbell=True)
                out[f"{d}_a{astar}_p{pplus}"] = {
                    "dict": d, "astar": astar, "pplus": pplus,
                    "ks_old_new": H.ks_distance(old, new),
                    "max_abs_diff": float(np.max(np.abs(old - new))),
                    "median_abs_H": float(np.median(np.abs(old)))}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, f'{v["ks_old_new"]:.5f}', f'{v["max_abs_diff"]:.2e}', f'{v["median_abs_H"]:.3g}')


if __name__ == "__main__":
    main()
