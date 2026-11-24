# Supplement S4 — per-fit dispersion, the fixed window, (x, x³) at α_c ∈ (2, 3), a locked interior β, RflyMAD in the run-3 regime, the scale-only reverse split

Written 2026-11-24. At that point the only S4 output was the design probe (training fits only) and
a smoke test of the two reproduction controls and of two edge cells (`poly3`, `E4 rev40`). No other
S4 window had been computed. It answers four questions raised in review of the manuscript that need computation:
1. the per-fit spread of the onset ratio and of the prediction error;
2. the dictionaries `(x, x³)` and `(x, x², x³)` at `α_c ∈ (2, 3)`;
3. RflyMAD in the run-3 regime, and a reverse split on clean laws that differ in scale only;
4. a locked interior-`β` cell.

It also gives the numerical check of the new Corollary `cor:detwindow` (stage-5 amendment
2026-11-24).

The thresholds below are registered here and are not moved after the run. Post-run notes go to a
dated section at the end. `DESIGN.md`, `DESIGN_S2.md` and `DESIGN_S3.md` are not touched.

## What is reused

- `h1v4.py`, `h1v4_s2.py`, `h1v4_s3.py` are imported and not modified. The replication functions
  `h1v4_s2.crossover_rep` and `h1v4_s3.crossover_rep` are called unchanged.
- Everything below comes from runs 2 and 3:
  - the detector, the fit, the winsorized comparator, the fitted sub-dictionary;
  - the window MC (`M = 500` pairs, 33-point grid up to `N_max = 31 623`);
  - the onset `N_on` (first `N` past the peak with `AUC_N ≤ 0.75`);
  - `R = 40` training replications per cell;
  - the LePage prediction;
  - training split `(0.05, 0.05)`, symmetric contaminant.
- New code: `h1v4_s4.py`, with the summary of a cell plus the per-fit values, their quartiles and
  the fixed-window check; `spec_s4.py`; `design_probe_s4.py`.

## Design probe

`design_probe_s4.py` → `results/design_probe_s4.json`. It runs 40 training fits per candidate and
computes no window AUC.

**Registration rule (blocks A and C):** a candidate is registered when its predicted onset
`λ_m^{α_c/(a*−α_c)} · median N*` lies in `[50, 10⁴]` and the share of fits with `μ̂ ≤ 0`
(`N* = 1`) is ≤ 0.10.

What the probe showed:
- **`(x, x³)` (`oo3`).** At `ε^te = 0.05 + 0.05` the crossover lies at `N ≥ 3·10⁵`. Only
  `0.2 + 0.2` brings it into the grid.
- **`(x, |x|³)` (`oe3`).** The sign of `K̂_t` is not yet a coin at `n ≤ 1000`:
  - `P̂{K̂_t > 0}` is 0.75–1.00;
  - the reason is the clean even difference in `Ŷ_t`, whose share decays only like
    `(nε)^{−1/α_c}`.
- **RflyMAD, `oe` dictionaries.** `K^sub_{l*}` is about 0.002, so the lock of `oe` says little
  there.
- **Scale-only `(0, 0.4)`.** It has `μ^te_L < 0` (every fit `N* = 1`), outside Theorem 7.

## Blocks

**S4-D (dispersion and the fixed window; 68 cells).** Every crossover cell of run 2 (30 cells,
block X) and of run 3 (38 cells, blocks A and B), with the same parameters and seeds.
- **Stored per cell, beyond the run-2/run-3 summary:**
  - the per-fit `K̂_t`, `N*`, `N_on`, `AUC_{N_max}` and its prediction;
  - quartiles and p90 of the onset ratio `N_on/(λ_m^{α_c/(a*−α_c)} N*)` (finite onsets);
  - quartiles and p90 of `|AUC_{N_max} − pred|`.
- **Fixed-window check.** At every grid `N ≥ 10³`, the mean over fits of `AUC_N(K̂)` against the
  mean over fits of the conditional prediction `Ψ(x_n) = Prob{1 + (N/N*)^{(a*−α_c)/α_c} sgn(K̂_t) Z > 0}`.
  This is the step `AUC_{N_n} − Ψ(x_n) → 0` of the proof of Cor. `cor:detwindow`.

**S4-A (two-column dictionaries inside the theory, `α_c ∈ {2.2, 2.5}`, `a_L = 1`, `a* = 3`).**
- Dictionaries `oo3 = (x, x³)` (same parity) and `oe3 = (x, |x|³)` (opposite parity).
- (B4)–(B5) hold: `2a_L = 2 < α_c < 3 = a*`, `a* + a_L = 4 > α_c`, and no exponent sum equals
  `α_c`.
- Clean laws as in runs 1–3.
- Registered cells (n, test split):
  - `oo3`, `α_c = 2.2`: `n = 200`, `(0.2, 0.2)`;
  - `oo3`, `α_c = 2.5`: `n = 200`, `(0.2, 0.2)`;
  - `oe3`, `α_c = 2.2`: `n = 200` at `(0.05, 0.05)` and `(0.1, 0)`; `n = 1000` at `(0.2, 0.2)` and
    `(0.4, 0)`;
  - `oe3`, `α_c = 2.5`: `n = 1000` at `(0.2, 0.2)` and `(0.4, 0)`.
- Reported only: `poly3 = (x, x², x³)` at `α_c ∈ {2.2, 2.5}`, `n = 200`, `(0.2, 0.2)`. Both `x²`
  and `x³` are inadmissible, so `|I| = 2` and no theorem of the paper applies.

**S4-B (a locked interior `β`).** `ee`, `α_c = 1.2`, `a* = 4`, `n = 1000`, run-3 layout.
- The probe gives lock 1.00 and `K^sub_{l*} = +0.209`.
- Test splits `(0.075, 0.025)` (`β^te = +½`) and `(0.025, 0.075)` (`β^te = −½`).
- Corollary 8(i) gives the limit `ρ(γ, −β^te)`: below ½ for `+½` (inversion), above ½ for `−½`.

**S4-C (RflyMAD clean pools in the run-3 regime).** Clean pools as in run 2 block D (train and
test flights disjoint).
- Pareto contaminant `α_c ∈ {1.2, 1.5}`, `a* ∈ {3, 4}`, run-3 layout.
- Test splits `(0.05, 0.05)` and `(0.10, 0)`.
- 18 cells registered by the rule:
  - `α_c = 1.2`, `a* = 3`:
    - `oe`, `n = 200`: sym, one;
    - `ee`, `n = 200`: sym, one;
    - `ee`, `n = 1000`: sym;
  - `α_c = 1.2`, `a* = 4`:
    - `oe` and `ee`, `n = 200`: sym, one;
    - `oe` and `ee`, `n = 1000`: sym, one;
  - `α_c = 1.5`:
    - `ee`, `a* = 3`, `n = 200`: sym;
    - `oe` and `ee`, `a* = 4`, `n = 200`: sym, one.

**S4-E (scale-only clean laws, forward and reverse one-sided splits).**
- Clean laws `N(0, 1.3²)` and `N(0, 1)`; `ee`, `α_c = 3`, `a* = 8`, `n = 200`.
- Test splits `(0.1, 0)`, `(0.4, 0)` (forward: the larger-scale class is contaminated) and
  `(0, 0.1)`, `(0, 0.4)` (reverse).
- `K^sub_{l*} = +0.170`, so Corollary 8(i) predicts inversion under the forward splits and none
  under the reverse ones.
- `(0, 0.4)` has `μ^te_L < 0` and is reported only.

## Controls (must pass before the grid; a miss aborts the run)

- **C14.** The S4 summary of the run-3 cell `oo`, `α_c = 1.2`, `a* = 3`, `n = 200`, sym, recomputed
  with its seeds, equals the stored values of `REPRO_KEYS` (difference ≤ 1e-9).
- **C15.** The same for the run-2 RflyMAD cell `ee`, `a* = 8`, `n = 200`, sym.

## Registered criteria

- **D0 — reproduction.** In all 68 D cells the recomputed `REPRO_KEYS` equal the stored run-2/3
  values (`reproMaxDiff ≤ 1e-9`).
- **D1 — per-fit onset ratio.**
  - Cells: those of Q2 (run 2: symmetric S2-A and S2-B, 14) and R2 (run 3: symmetric S3-A, 20).
  - Requirement: the per-fit ratio has `q25 ≥ 0.5` and `q75 ≤ 2.0` in ≥ 80 % of these 34 cells.
- **D2 — per-fit prediction error.**
  - Cells: those of Q3 (run 2: one-sided, 12) and R3 (run 3: one-sided S3-A and all of S3-B, 18).
  - Requirement: p90 of `|AUC_{N_max} − pred|` ≤ 0.15 in ≥ 80 % of these 30 cells.
- **D3 — fixed window.** In ≥ 80 % of the 34 D1 cells,
  `max_{N ≥ 10³} |mean AUC_N − mean Ψ| ≤ 0.05`.
- **A1 — collapse and onset, theory dictionaries.**
  - Cells: the five symmetric A cells.
  - Requirements, as R1 and R2:
    - median peak ≥ 0.80;
    - `N_on` finite in ≥ 80 % of the fits;
    - median `N_on` ≤ 2·10⁴;
    - median onset ratio in `[0.67, 1.5]`.
- **A2 — past the crossover.** In the three one-sided A cells, median `|AUC_{N_max} − pred|`
  ≤ 0.10.
- **B1 — locked interior `β`.** In both B cells:
  - lock ≥ 0.95;
  - median `|AUC_{N_max} − pred|` ≤ 0.10;
  - median `AUC_{N_max}` < ½ at `β^te = +½` and > ½ at `β^te = −½`.
- **C1 — RflyMAD, symmetric cells (10).** R1 and R2 as in A1.
- **C2 — RflyMAD, one-sided cells (8).** Median `|AUC_{N_max} − pred|` ≤ 0.10.
- **C3 — RflyMAD winsorizing (symmetric cells).**
  - Median `AUC_{N_max}(K̂^W)` ≥ 0.90.
  - Median `AUC₁₀(K̂^W) − AUC₁₀(K̂^sub)` ≥ −0.02.
- **E1 — scale-only splits.** In `(0.1, 0)`, `(0.4, 0)` and `(0, 0.1)`, median
  `|AUC_{N_max} − pred|` ≤ 0.10.
- **Reported, not criteria:**
  - lock, `P̂{K̂_t > 0}` and failed fits in every cell;
  - the `poly3` cells;
  - `(0, 0.4)` of S4-E;
  - the winsorized `AUC_{N_max}` under the scale-only splits;
  - the quartiles of every D cell, including RflyMAD and those outside D1/D2;
  - the onset ratio of one-sided cells.

A criterion that fails is a finding, recorded with its cell, not a reason to change the grid.

## Post-run notes (2026-11-24)

Run `results/h1v4-s4-2026-11-24/`:
- 102 cells, 5369 s, no failed replication.
- Controls: C14 and C15 reproduce the stored cells exactly (difference 0).
- Verdicts by `criteria_s4.py`:

| criterion | verdict | detail |
|---|---|---|
| D0 reproduction | PASS | 68/68, every difference 0 |
| D1 per-fit onset ratio | PASS | 34/34. The per-fit `q25` is ≥ 0.80 and the `q75` ≤ 1.31 in every cell except run-2 `oe`, `n = 50` (`q75` 1.85) |
| D2 per-fit prediction error | PASS | 30/30; p90 ≤ 0.038 |
| D3 fixed window | PASS | 34/34; `max_{N≥10³} |mean AUC_N − mean Ψ|` ≤ 0.011 |
| A1 theory dictionaries, symmetric | **FAIL** | 0/5 (details below) |
| A2 theory dictionaries, one-sided | PASS | 3/3; error 0.000. Most fits have `K̂_t > 0`, `β^te = +1`, limit 1: no collapse to measure |
| B1 locked interior `β` | PASS | lock 1.00 in both cells. `β^te = +½`: 0.254 observed, 0.249 predicted (inversion). `β^te = −½`: 0.784 against 0.781 |
| C1 RflyMAD, symmetric | **FAIL** | 7/10. Every ratio is 0.91–1.06 and every peak ≥ 0.95. The failures are on the "onset found in 80 %" clause only: `ee`, `α_c = 1.2`, `a* = 3`, `n = 1000` (0.70); `ee`, `α_c = 1.5`, `a* = 3`, `n = 200` (0.72); `oe`, `α_c = 1.2`, `a* = 4`, `n = 1000` (0.75). As R1 in run 3, median `N*` is 1.2–1.8·10⁴, so part of the fits cross beyond `N_max` |
| C2 RflyMAD, one-sided | PASS | 8/8; error ≤ 0.006 |
| C3 RflyMAD, winsorizing | PASS | 10/10; winsorized `AUC_{N_max}` = 1.000; clean cost ≥ −0.001 |
| E1 scale-only splits | PASS | 3/3; error 0.000 |

**A1 failure.** For `(x, x³)` and `(x, |x|³)` at `α_c ∈ (2, 3)` the crossover length is not
determined by the training sample to within orders of magnitude.
- The exponent of `μ̂/|K̂_t|` in `N*` is `α_c/(a*−α_c)` = 2.75 (`α_c = 2.2`) or 5 (`α_c = 2.5`).
  Across the 40 fits of a cell the interquartile range of `N*` spans 1.7–5.7 decades.
- An onset inside the grid is therefore found in only 23–48 % of the fits. The median peak is
  0.79–0.95.
- The 40-fit medians of the design probe were themselves unstable: the run medians are 1.3–24 times
  the probe's.
- Per-fit onset ratios differ by parity:
  - same parity `oo3`: median 0.74 at `α_c = 2.2` (IQR 0.52–0.95) and 2.10 at `α_c = 2.5`
    (IQR 0.14–9.2);
  - opposite parity `oe3`: 2.6–91.
- The excess comes from the term that the run-1 `oe` outlier isolated: the clean drift of the top
  column, `K̂_t` times the clean difference of `φ_t` between the classes. `N*_n(K̂)` drops this
  term, and at these `n` it is not small.
  - Check: `diag_s4_clean_drift.py` → `results/diag_s4_clean_drift.json`. It is post-hoc, not
    registered, and was added during review.
  - The refit with the run seeds reproduces every stored per-fit `N*`.
  - With `μ_eff = μ̂ + K̂_t·Δ_clean` the median onset ratio is 0.60–1.11 in all five cells:
    `oe3` 0.90 / 0.77 / 1.11, `oo3` 0.77 / 0.60.
  - The spread of `N*` across fits explains the censoring (onset found in only 23–48 % of fits),
    not the ratio, whose denominator is each fit's own `N*`.
- The conditional limit itself holds in the same cells. The median error is 0.013–0.053 in the
  symmetric cells and 0.000 in the one-sided ones. The fixed-window difference is ≤ 0.061.
- A1 fails as registered. What fails is the use of `N*` to place the onset, not the limit.

**Scale-only splits (E), per fit.** At `n = 200` the `ee` sign is not locked (40–53 % of fits have
`K̂_t > 0`, `K^sub_{l*} > 0`). Split by sign:

| split | locked fits (`K̂_t < 0`): median `AUC_{N_max}` | unlocked fits: median `AUC_{N_max}` |
|---|---|---|
| `(0.1, 0)` | 0.35 (19 fits) | 1.00 |
| `(0.4, 0)` | 0.001 (16 fits) | 1.00 |
| `(0, 0.1)` | 1.00 | 0.038 |
| `(0, 0.4)` | 1.00 | 0.000 |

- The locked fits behave as Cor. 8(i) says: they invert under the forward split and do not under the
  reverse one.
- In `(0, 0.4)` the admissible drift is negative (`μ^te_L = −0.0068`). The fitted sub-dictionary
  then has `AUC_{N_max}` = 0.00, and the winsorized detector has 0.53. This is the case of the
  remedy remark, now seen.

**`poly3` (reported).**
- `α_c = 2.2`: median peak 0.84; onset found in 78 % of fits (median 880).
- `α_c = 2.5`: median peak 0.92; onset found in 20 % of fits.
- In both, `K̂` of `x²` is positive in 97–100 % of fits; `K̂` of `x³` is positive in 38–40 %.
- Winsorizing gives 1.000 in both.
