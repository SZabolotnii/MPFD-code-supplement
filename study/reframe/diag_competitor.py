"""Post-run diagnostic D1 (2026-11-22): why P4 (fair coin) failed for `oe`.

Not a registered criterion and not part of the mc_kernel run. Hypothesis: in `oe` the lower
admissible column |x|^0.5 has the top column's parity (both even), so its cross term
sum |x|^{0.5+a*} has a deterministic sign and is only b_n^{-(a_L - a_l)} = b_n^{-1/2} smaller
than the mean-zero l* term. Test: drop that column (dictionary {x, |x|^a*}) and re-measure
P(K_t > 0) at the registered n and R, same splits; the fair coin should then appear.
Also reports the same frequency for the full `oe` dictionary on the same draws.
Writes results/diag_competitor.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h1v4 as H  # noqa: E402

SPLITS = {"bal": (0.05, 0.05), "plus": (0.10, 0.0), "minus": (0.0, 0.10)}


def main() -> int:
    out = {}
    for astar in (2.4, 4.0):
        full = H.columns("oe", astar)
        reduced = full[1:]
        for s, split in SPLITS.items():
            rng = np.random.default_rng(H.seed_of({"diag": "competitor", "astar": astar, "split": s}))
            pos_full, pos_red = [], []
            for _ in range(H.R_A):
                xp = H.draw(rng, H.NS_A[-1], split[0], H.G_PLUS, 0.5)
                xm = H.draw(rng, H.NS_A[-1], split[1], H.G_MINUS, 0.5)
                pos_full.append(H.fit(full, xp, xm)[0][-1] > 0)
                pos_red.append(H.fit(reduced, xp, xm)[0][-1] > 0)
            key = f"oe_a{astar}_{s}"
            out[key] = {"pos_full": float(np.mean(pos_full)), "pos_reduced": float(np.mean(pos_red)),
                        "within_P4_band_reduced": bool(abs(np.mean(pos_red) - 0.5) <= 0.087)}
            print(key, out[key], flush=True)
    path = Path(__file__).resolve().parent / "results" / "diag_competitor.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
