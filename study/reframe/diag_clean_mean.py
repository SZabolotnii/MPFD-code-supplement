"""Post-run diagnostic D2 (2026-11-22): the clean top-column mean difference behind the P4 failure.

Not a registered criterion. D1 (diag_competitor.py) removed the lower even column of `oe` and
P(K_t > 0) moved towards 1, not 1/2. Remaining candidate: the clean part of Y_t,
E_{G+}|x|^a* - E_{G-}|x|^a*, which is O(1) and positive because G+ has the larger scale; the
theorem neglects it as o(b_n^{a*+a_L}/n), but that ratio decays only like n^{1-(a*+a_L)/3}.
Test: the reduced dictionary {x, |x|^a*} with equal clean scales, G+ = N(0.3, 1) (the clean
difference of |x|^a* then comes from the 0.3 shift only, and is much smaller). Same n, R.
Writes results/diag_clean_mean.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1v4 as H  # noqa: E402

SPLITS = {"bal": (0.05, 0.05), "plus": (0.10, 0.0), "minus": (0.0, 0.10)}
G_PLUS_EQ = (0.3, 1.0)


def main() -> int:
    out = {}
    for astar in (2.4, 4.0):
        reduced = H.columns("oe", astar)[1:]
        ydiff = (H._gauss_moment("e", astar, *H.G_PLUS) - H._gauss_moment("e", astar, *H.G_MINUS),
                 H._gauss_moment("e", astar, *G_PLUS_EQ) - H._gauss_moment("e", astar, *H.G_MINUS))
        for s, split in SPLITS.items():
            rng = np.random.default_rng(H.seed_of({"diag": "clean_mean", "astar": astar, "split": s}))
            pos = []
            for _ in range(H.R_A):
                xp = H.draw(rng, H.NS_A[-1], split[0], G_PLUS_EQ, 0.5)
                xm = H.draw(rng, H.NS_A[-1], split[1], H.G_MINUS, 0.5)
                pos.append(H.fit(reduced, xp, xm)[0][-1] > 0)
            key = f"oe_reduced_eqscale_a{astar}_{s}"
            out[key] = {"pos": float(np.mean(pos)), "within_P4_band": bool(abs(np.mean(pos) - 0.5) <= 0.087),
                        "clean_Ydiff_registered_laws": ydiff[0], "clean_Ydiff_equal_scale": ydiff[1]}
            print(key, out[key], flush=True)
    path = Path(__file__).resolve().parent / "results" / "diag_clean_mean.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
