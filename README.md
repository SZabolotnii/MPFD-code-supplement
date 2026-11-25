# MPFD Code Supplement

Reproducibility code for two papers:

1. **Where the contamination enters decides how a Fisher moment detector fails**
   (S. Zabolotnii, 2026) -- `study/reframe/`, `paper/reframe/`, `lean/MCP/Reframe.lean`;
   see the section "Contamination paper" below.
2. **Mixed-Parity Fractional Dictionaries for Moment-Based Discrimination** (S. Zabolotnii, 2026),
   a Short Communication -- everything else in this repository.

## Contamination paper (`study/reframe/`)

Every number in the paper is a macro of `paper/reframe/generated/numbers.tex` or a cell of a
generated table, and `study/reframe/make_tex_reframe.py` builds them from the stored results only;
`study/tests/test_reframe_generated.py` fails if the committed fragments are not what the stored
results give. The figures come from `make_figs_reframe.py`.

| paper | what | files |
|---|---|---|
| simulation run 1 (Blocks A, B) | rate and sign of the inadmissible coefficient, clean-window limit, onset of the collapse | `spec_h1v4.py`, `h1v4.py`, design and registered criteria in `DESIGN.md`, results in `results/h1v4-2026-11-22/`, verdicts by `criteria.py` |
| simulation run 2 | the collapse at practical window lengths, the conditional limit, winsorizing; Student-t3 contaminant | `spec_s2.py`, `h1v4_s2.py`, `DESIGN_S2.md`, `results/h1v4-s2-2026-11-22/`, `criteria_s2.py` |
| simulation run 3 | heavier contaminants (`alpha_c` 1.2 and 1.5), interior test skewness and the reverse one-sided split, the training-size rate in two-column dictionaries with equal class scales | `spec_s3.py`, `h1v4_s3.py`, `design_probe_s3.py`, `DESIGN_S3.md`, `results/h1v4-s3-2026-11-23/`, `criteria_s3.py` |
| simulation run 4 | runs 2 and 3 recomputed with their seeds and per-fit values (dispersion, fixed-window check), `(x, x^3)` and `(x, |x|^3)` at `alpha_c` 2.2 and 2.5, a locked interior-skewness cell, RflyMAD in the run-3 regime, scale-only clean laws | `spec_s4.py`, `h1v4_s4.py`, `design_probe_s4.py`, `DESIGN_S4.md`, `results/h1v4-s4-2026-11-24/`, `criteria_s4.py` |
| simulation run 5 | the limit law of the fixed-window corollary (run-3 two-column fits, exact AUC from the characteristic function at window lengths up to 10^17), and `(x, |x|^3)` under one-sided splits that make the fits collapse; LePage sampler with its tail restored | `spec_s5.py`, `h1v4_s5.py`, `design_probe_s5.py`, `DESIGN_S5.md`, `results/h1v4-s5-2026-11-24/`, `criteria_s5.py` |
| semi-synthetic demonstration | RflyMAD clean laws with a synthetic Pareto contaminant | `h1v4_s2.py` (`rfly_pools`), cells `src=rfly` of runs 2 and 4 |
| post-hoc diagnostics | clean drift of the top column, onset censoring, population `K^sub`, competitor, general-`p_+` sign limit, sampler of `H` | `diag_*.py` -> `results/diag_*.json` (not registered; the paper says so) |
| algebraic lemmas | bordered solve, parity of products, sign of a same-parity sum, positivity parameter, crossover balance | `lean/MCP/Reframe.lean` |

Each run directory holds one JSON file per cell, `MANIFEST.sha256` and `RUN.json`. The runs were
made with `mc_kernel.py` (vendored here; it runs the controls first and stops if one fails):

```bash
cd study/reframe
python3 mc_kernel.py spec_h1v4.py --resume   # hours; cells already present and hashed are reused
python3 mc_kernel.py spec_s2.py --resume
python3 mc_kernel.py spec_s3.py --resume     # about one hour
python3 mc_kernel.py spec_s4.py --resume     # about 1.5 hours
python3 mc_kernel.py spec_s5.py --resume     # about 3 hours (block F: exact AUC for 1800 fitted vectors)
python3 criteria.py && python3 criteria_s2.py && python3 criteria_s3.py
python3 criteria_s4.py && python3 criteria_s5.py
python3 make_tex_reframe.py && python3 make_figs_reframe.py
```

The RflyMAD cells and two diagnostics need the per-flight series of the no-fault real flights of
RflyMAD (doi:10.1177/02783649241305153). The dataset is copyrighted by the Reliable Flight Control
Group, Beihang University, for non-commercial use, and is not redistributed here. Point
`MPFD_RFLY_CACHE` at the extracted file or place it at `study/reframe/data/ext_cache_f.npz`.
Without it those cells cannot be recomputed; their stored results are included.

## Short Communication

This repository is deliberately separate from the manuscript repository. It contains the study
library, the seven pre-registered checks with their frozen JSON results, the exploratory pilots
that preceded the last of them, the generators that turn those results into the figures and the
numeric macros of the paper, the value gate, and the Lean 4 formalisation. It contains no
manuscript text.

## What the paper claims, and which script establishes it

| claim in the paper | check | script |
|---|---|---|
| Prop. 1: half-line decomposition of the deflection in one-sided fractional moments; `M(A) = H(A, A)`; the classical quadratic rule as a sub-dictionary; `Ỹ²/S` increment (Prop. 3) on a concrete pair | LC1 | `study/checks/l01_halfline_identity.py` |
| Prop. 1(b): an all-odd dictionary is blind to a symmetric pair, an all-even one to a mirror-skew pair; the diagnostic curves `Δν(q)`, `Δσ(q)` of **Fig. 1(a)** | LC2 | `study/checks/l02_parity.py` |
| Prop. 2: admissibility `2·a_max < α_c` under `ε`-contamination; drift of the classical statistic versus convergence of `M`; **Fig. 2** (AUC versus `ε`); Hill ceiling `α̂_c/2` | LC3 | `study/checks/l03_admissibility.py` |
| **Table 1** and **Fig. 1(b)**: captured fraction `J/J_∞` at a three-column budget for `H`, `M`, `P`, `E` over the seven-pair panel; gains of `H` over `M` and over `E`; **Table S3** (where the optima sit, grid-edge cells marked) | LC4 | `study/checks/l04_split_vs_parity.py` |
| Exponent selection: oracle free odd pair versus the transition-polynomial path; mixed parity versus `P`; the feasible arm (**Table S4**) | LC5 | `study/checks/l05_exponent_selection.py` |
| **Table S2**: real-data pair (Tampere drone recordings), two time splits, moving-block bootstrap, `n_recordings = 1` per class | LC6 | `study/checks/l06_tampere.py` |
| **Table S5**: real sea clutter (IPIX Dartmouth 1993), clutter-only cells versus the documented target cell, leave-one-file-out over 14 recordings, the dictionary chosen on the training files by `J` alone | LC7 | `study/checks/l07_ipix.py` |
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
python3 checks/l07_ipix.py                      # ~40 min (14 folds x 4 grids)
python3 make_figs.py && python3 make_tex.py     # paper/figs, paper/generated
python3 verify_reported_values.py               # PASS iff the committed fragments are current
```

Re-running a check overwrites `results/l0N.json`; compare with `git diff` to see that the
numbers reproduce (floating-point noise in the last digits is expected across platforms).

`l07_ipix.py` needs the 14 stare-mode files of the public McMaster IPIX Dartmouth 1993
database (<https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html>, the ones listed as
carrying a target); point `MPFD_IPIX_RAW` at the directory holding them, or place them under
`$RESEARCH_DATA_ROOT/store/ipix-dartmouth-1993/data`. The 8-bit ADC samples are **unsigned**:
read as `int8` the clutter shows a kurtosis near 114, which is wrap-around and not sea clutter --
`mcp/data/ipix.py` views them as `uint8` and centres at 127.5. Without the files the check is
skipped; the frozen `results/l07.json` is included.

`study/pilots/` holds the three exploratory pilots that preceded LC7, each with its pass criteria
written down before its code: partial-discharge antenna signals and CWRU bearing vibration, both
of which failed the gain criterion and are **not** used in the manuscript, and the IPIX pilot that
became LC7. The PD failure is instructive rather than incidental -- the mechanism is fully present
there, yet the classical control still wins, because the heavy tail is the class difference itself
rather than a contaminant common to both classes, so the deflection criterion and AUC rank the
exponents in opposite directions.

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
