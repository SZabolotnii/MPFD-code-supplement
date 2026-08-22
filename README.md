# MPFD Code Supplement

Reproducibility code for the Short Communication

**Mixed-Parity Fractional Dictionaries for Moment-Based Discrimination** (S. Zabolotnii, 2026)

This repository is deliberately separate from the manuscript repository. It contains the study
library, the six pre-registered checks with their frozen JSON results, the generators that turn
those results into the figures and the numeric macros of the paper, the value gate, and the
Lean 4 formalisation. It contains no manuscript text.

## What the paper claims, and which script establishes it

| claim in the paper | check | script |
|---|---|---|
| Prop. 1: half-line decomposition of the deflection in one-sided fractional moments; `M(A) = H(A, A)`; the classical quadratic rule as a sub-dictionary; `Ỹ²/S` increment (Prop. 3) on a concrete pair | LC1 | `study/checks/l01_halfline_identity.py` |
| Prop. 1(b): an all-odd dictionary is blind to a symmetric pair, an all-even one to a mirror-skew pair; the diagnostic curves `Δν(q)`, `Δσ(q)` of **Fig. 1(a)** | LC2 | `study/checks/l02_parity.py` |
| Prop. 2: admissibility `2·a_max < α_c` under `ε`-contamination; drift of the classical statistic versus convergence of `M`; **Fig. 2** (AUC versus `ε`); Hill ceiling `α̂_c/2` | LC3 | `study/checks/l03_admissibility.py` |
| **Table 1** and **Fig. 1(b)**: captured fraction `J/J_∞` at a three-column budget for `H`, `M`, `P`, `E` over the seven-pair panel; gains of `H` over `M` and over `E`; **Table S3** (where the optima sit, grid-edge cells marked) | LC4 | `study/checks/l04_split_vs_parity.py` |
| Exponent selection: oracle free odd pair versus the transition-polynomial path; mixed parity versus `P`; the feasible arm (**Table S4**) | LC5 | `study/checks/l05_exponent_selection.py` |
| **Table S2**: real-data pair (Tampere drone recordings), two time splits, moving-block bootstrap, `n_recordings = 1` per class | LC6 | `study/checks/l06_tampere.py` |
| Prop. 1(b) and Prop. 3, algebraic core (**Table S1**) | — | `lean/MCP/Parity.lean`, `lean/MCP/Schur.lean`, audit in `lean/Audit.lean` |

Registered thresholds and the dated post-run annotations are kept in separate sections of
`study/CHECKS.md`; thresholds were never moved after registration. Four sub-criteria returned
`PASS = False` and are discussed there and in the paper.

`study/mcp/` is the shared core: panel laws (`laws.py`, `panel.py`), one-sided fractional
moments and the Hill estimator (`moments.py`), the dictionaries (`basis.py`), the deflection
projection `F K = Y` with `F = C^(m) + C^(n)` (`projection.py`), and window scoring / AUC
(`eval.py`).

## Every number in the paper comes from here

* `study/results/l0N.json` are the frozen outputs of the checks (seed `20260821`, provenance in
  `_meta`).
* `study/make_tex.py` emits `paper/generated/{numbers,table1..table4}.tex` from those files;
  `study/make_figs.py` emits `paper/figs/`. The paper cites macros from these fragments only.
* `study/verify_reported_values.py` regenerates the fragments into a temporary directory and
  fails if they differ from the committed ones; against the manuscript it also rejects any
  numeric literal that no macro backs (here the manuscript is absent, so only the staleness
  check is exercised).
* `study/tests/test_make_tex.py` pins by hand a set of macro values computed independently from
  the JSON files, so a wrong-but-deterministic JSON-to-macro mapping cannot pass.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
cd study
python3 -m pytest tests -q                      # library identities, pinned macros, gate probes
python3 checks/l01_halfline_identity.py         # seconds
python3 checks/l02_parity.py                    # ~1 min
python3 checks/l03_admissibility.py             # minutes (n up to 1e6, 30 seeds)
python3 checks/l04_split_vs_parity.py           # ~1 min
python3 checks/l05_exponent_selection.py        # minutes
python3 make_figs.py && python3 make_tex.py     # paper/figs, paper/generated
python3 verify_reported_values.py               # PASS iff the committed fragments are current
```

Re-running a check overwrites `results/l0N.json`; compare with `git diff` to see that the
numbers reproduce (floating-point noise in the last digits is expected across platforms).

`l06_tampere.py` needs the raw IQ recordings of the public Tampere dataset
(doi:10.5281/zenodo.4264467); point `MPFD_TAMPERE_RAW` at the directory holding
`Parrot_mambo_control_2G` and `Parrot_mambo_video_2G`. Without it, the check and
`tests/test_tampere.py` are skipped; the frozen `results/l06.json` is included.
`tests/test_bridge_hl.py` cross-checks the one-sided moments against a sibling engine that is
not public and is skipped automatically.

## Lean

```bash
cd lean
lake exe cache get        # mathlib oleans for the pinned revision
lake build                # 0 sorry
lake env lean Audit.lean  # every theorem depends on propext, Classical.choice, Quot.sound only
```

Toolchain `leanprover/lean4:v4.26.0`, mathlib pinned in `lake-manifest.json`.

## Software versions used for the frozen results

Python 3 with numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.9 (recorded per check in `_meta`).

## License

MIT (see `LICENSE`). If you use this code, please cite the paper (see `CITATION.cff`).
