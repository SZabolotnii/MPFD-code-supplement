"""Design-time probe for DESIGN_S2.md (run before any S2 test window exists).

Uses training fits only: for each candidate cell, the median of N*_n(K-hat) over 40 fits,
and the onset factor lambda_m^(alpha/(a*-alpha)) of the crossover theorem. No AUC is
computed here, so the cell selection cannot depend on an S2 outcome.

Output: results/design_probe_s2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1v4 as h  # noqa: E402

SPLIT, PP = (0.05, 0.05), 0.5


def main() -> None:
    rng = np.random.default_rng(20261122)
    rows = []
    for name in ("oo", "oe", "ee", "eo"):
        for astar in (5.0, 6.0, 8.0):
            cols = h.columns(name, astar)
            for n in (50, 200, 1000):
                for ete in (0.2, 0.4):
                    ns = []
                    for _ in range(40):
                        K, _ = h.fit(cols, h.draw(rng, n, SPLIT[0], h.G_PLUS, PP),
                                     h.draw(rng, n, SPLIT[1], h.G_MINUS, PP))
                        ns.append(h.nstar(cols, K, (ete / 2, ete / 2), PP))
                    ns = np.array(ns)
                    rows.append({"dict": name, "astar": astar, "n": n, "eps_te_total": ete,
                                 "medNstar": float(np.median(ns)),
                                 "q25": float(np.quantile(ns, 0.25)),
                                 "q75": float(np.quantile(ns, 0.75))})
    gam = h.ALPHA / 8.0
    lam = h.lambda_mid(gam, 0.75)
    out = {"rows": rows, "gamma_a8": gam, "lambda_m_a8": lam,
           "onset_factor_a8": lam ** (h.ALPHA / (8.0 - h.ALPHA))}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "design_probe_s2.json").write_text(json.dumps(out, indent=1))
    print("onset factor at a*=8:", out["onset_factor_a8"])
    for r in rows:
        if r["astar"] == 8.0 and r["eps_te_total"] == 0.4:
            print(r)


if __name__ == "__main__":
    main()
