# Supplement S3 — heavier contaminants, interior test skewness, the reverse split, the training rate

Written 2026-11-23, before any S3 window or fit beyond the design probe exists. The author asked
for this supplement after round 1 of the fourth manuscript review, which objected that:
- every run fixes `α_c = 3`, so run 2 needed `a* = 8` and `ε^te = 0.4` to reach practical `N`;
- `β^te` is only ever 0 or 1, and the reverse one-sided split was never run;
- the training-size rate is never seen, even in a design with small remainders.

The thresholds below are registered here and are not moved after the run. Post-run notes go to a
dated section at the end. `DESIGN.md` and `DESIGN_S2.md` are not touched.

## What is reused

- The detector, the fit, the winsorized comparator (0.5 % / 99.5 % training quantiles), the fitted
  sub-dictionary, the window MC (`M = 500` pairs, the 33-point grid up to `N = 31 623`), the onset
  `N_on` (first `N` past the peak with `AUC_N ≤ 0.75`), the clean-window `AUC₁₀` (20 000 pairs) and
  the conditional prediction by LePage sampling: all from `h1v4_s2.py`, unchanged.
- Clean laws `G₊ = N(0.3, 1.3²)`, `G₋ = N(0, 1)` in blocks A and B.
- Training contamination `ε^tr = (0.05, 0.05)`, symmetric contaminant `p₊ = ½`, `R = 40` training
  replications per crossover cell.
- New code: `h1v4_s3.py` passes `α_c` explicitly (`Pareto(alpha)`, `nstar`, `predict`,
  `crossover_cell`, `rate_block`); `spec_s3.py` is the run.

## Design probe

`design_probe_s3.py` → `results/design_probe_s3.json`. Training fits only (40 per cell), no window
AUC. It reports the median `N*_n(K̂)`, the population `K^sub_{l*}` and `μ^te_L`, and the predicted
onset `λ_m^{α_c/(a*−α_c)} · median N*`.
- Block A: at `n = 200` the predicted onset lies in 480–3700 for every (dictionary, `α_c`, `a*`)
  and both test splits; at `n = 1000` it lies inside the grid only for `α_c = 1.2`, `a* = 4`
  (3300–8600), elsewhere it reaches 1.2·10⁴–1.7·10⁵.
- Block B: predicted onset 250–9100 in all eight cells; `μ^te_L > 0` in the population for all
  of them, and `μ̂ ≤ 0` (so `N* = 1`) in 10 % of the fits of `oe` and 3 % of `ee` under `(0, 2/5)`.

## Blocks

**S3-A (heavier contaminants, realistic contamination).**
- Pareto contaminant with `α_c ∈ {1.2, 1.5}` (`c₀ = 1`).
- Dictionaries `oo, oe, ee, eo` with the run-1 layout and the lower exponents scaled by `α_c/3`
  (so `0.5, 1, 1.2` become `0.2, 0.4, 0.48` at `α_c = 1.2` and `0.25, 0.5, 0.6` at `α_c = 1.5`);
  top exponent `a* ∈ {3, 4}`. (B4) and (B5) hold: `2a_l < α_c`, `a* + a_L > α_c`, no exponent sum
  equals `α_c`.
- Test contamination: symmetric `(0.05, 0.05)`; one-sided `(0.10, 0)` for the even tops.
- Cells: `n = 200` for every combination (16 symmetric, 8 one-sided), plus `n = 1000` at
  `α_c = 1.2`, `a* = 4` (4 symmetric, 2 one-sided): 30 cells.

**S3-B (interior `β^te` and the reverse split), `α_c = 3`, `a* = 8`, `n = 200`, `oe` and `ee`.**
- Test splits `(0.3, 0.1)` (`β^te = +½`), `(0.1, 0.3)` (`β^te = −½`), `(0, 0.1)` and `(0, 0.4)`
  (`β^te = −1`, the reverse one-sided split): 8 cells.
- With `K^sub_{l*} > 0`, the reverse split has `sgn(K^sub_{l*}) β^te < 0`: by Corollary 8(i) the
  same-parity detector does not invert, its limit is `ρ(γ, 1) = 1`.

**S3-C (training rate, two columns, equal class scales), `α_c = 3`.**
- Dictionaries `oo4 = (sgn(x)|x|, sgn(x)|x|⁴)` (same parity) and `oe4 = (sgn(x)|x|, |x|⁴)`
  (opposite parity); `a_L = 1`, `a* = 4`.
- Clean laws `N(0.3, 1)` and `N(0, 1)`: equal scales, so no even clean difference; one lower column,
  so no lower-column remainder.
- Split `(0.05, 0.05)`, `n ∈ {2000, 8000, 32000, 128000, 512000}`, `R = 300`.
- Measured: median `|K̂_t|` per `n` and its log-log slope over the three largest `n`, lock frequency,
  `P̂{K̂_t > 0}`, and the Kolmogorov distance of `−b_n^{a*−a_L} K̂_t / K^sub_{l*}` at the largest `n`
  to the law of `H` (sampler with the full tail variance, `campbell=True`).

## Controls (must pass before the grid; a miss aborts the run)

- **C11, Pareto(α) sampler.** For `α ∈ {1.2, 1.5}`, from 4·10⁶ draws: `P(|C| > x)` at
  `x ∈ {10, 100}` within 4 SE of `x^{−α}`; `E|C|^q` at `q ∈ {0.2, 0.5}` within 4 SE of
  `α/(α − q)`.
- **C12, stable distribution function at the new `γ`.** For `γ ∈ {0.3, 0.375, 0.4, 0.5}`,
  `h1v4.stable_cdf_sym` agrees with the empirical CDF of 2·10⁵ LePage draws within 0.005 at three
  points.
- **C13, the S3 engine reproduces run 2 at `α_c = 3`.** One replication of the `ee`, `a* = 8`,
  `n = 200`, symmetric cell with the same seed gives the same `K̂_t`, `N*`, `N_on` and `AUC` at
  `N_max` through `h1v4_s3.crossover_rep` and `h1v4_s2.crossover_rep` (difference ≤ 1e-12).

## Registered criteria

- **R1 — collapse at practical `N` (S3-A, symmetric cells).** As Q1: median peak ≥ 0.80; `N_on`
  finite in ≥ 80 % of replications; median `N_on` ≤ 2·10⁴.
- **R2 — onset formula (S3-A, symmetric cells).** As Q2: median `N_on / (λ_m^{α_c/(a*−α_c)} N*)`
  in `[0.67, 1.5]`.
- **R3 — limit past the crossover (S3-A one-sided cells and all of S3-B).** As Q3: median
  `|AUC_{N_max}(K̂) − pred|` ≤ 0.10, with `pred` the conditional prediction at `c = N_max/N*`.
- **R4 — winsorizing (S3-A, symmetric cells).** As Q5: (a) median `AUC_{N_max}(K̂^W)` ≥ 0.90;
  (b) median `AUC₁₀(K̂^W) − AUC₁₀(K̂^sub)` on clean windows ≥ −0.02.
- **R5 — the rate (S3-C).** In both dictionaries the slope is within 0.10 of `−(a* − a_L)/α_c = −1`.
- **R6 — the law of `H` (S3-C, `oo4`).** Lock frequency ≥ 0.95 at `n = 512 000` and Kolmogorov
  distance to `H` ≤ 0.08 (the 5 % critical value for 300 replications).
- **R7 — the opposite-parity coin (S3-C, `oe4`).** `|P̂{K̂_t > 0} − ½|` ≤ 0.087 (3 SE at `R = 300`)
  at `n = 512 000`.
- **Reported, not criteria:** the winsorized `AUC_{N_max}` under the reverse splits of S3-B (does
  clipping invert when the contaminated class has the smaller clean scale?); lock and
  `P̂{K̂_t > 0}` in every crossover cell; `cond(F̂)`; failed replications per cell.

A criterion that fails is a finding, recorded with its cell, not a reason to change the grid.

## Post-run notes (2026-11-23)

Run `results/h1v4-s3-2026-11-23/`: 40 cells, 3206 s, no failed replication. Controls: C11 max |z| =
1.88; C12 max error 0.0021; C13 identical to run 2 (difference 0). Verdicts by `criteria_s3.py`:

| criterion | verdict | detail |
|---|---|---|
| R1 collapse at practical `N` | **FAIL** | 18/20. Median peak ≥ 0.89 and median `N_on` 463–5490 in all 20. Failures on the "finite in 80 %" clause only: `oe`, `α_c = 1.2`, `a* = 4`, `n = 1000` (0.72) and `oe`, `α_c = 1.5`, `a* = 3`, `n = 200` (0.78); in both the median `N*` (1.6·10⁴, 5.5·10³) puts part of the fits beyond `N_max` |
| R2 onset formula | PASS | 20/20; median `N_on / (λ_m^{α_c/(a*−α_c)} N*)` = 0.91–1.06 |
| R3 limit past the crossover | PASS | 18/18; median `|AUC_{N_max} − pred|` ≤ 0.017. `β^te = +½`: 0.77 observed, 0.77 predicted; `β^te = −½`: 0.23–0.27 against 0.23–0.24; reverse splits: `ee` stays at 1.00 (no inversion, as Cor. 8(i) predicts), `oe` goes to 0.00–0.16 by its sign split |
| R4 winsorizing | PASS | 20/20; winsorized `AUC_{N_max}` = 1.000 in every S3 cell, the reverse splits included; clean `AUC₁₀` cost ≥ −0.002 |
| R5 rate, two columns | PASS | slopes −0.946 (`oo4`) and −0.942 (`oe4`) against −1 |
| R6 law of `H` | PASS | lock 1.00 at `n = 512 000`; Kolmogorov distance 0.068 |
| R7 opposite-parity coin | PASS | `P̂{K̂_t > 0}` = 0.52 at `n = 512 000` (0.50–0.52 at every `n`) |

- The one-sided cells of block A and the interior/reverse cells of block B have onset ratios of
  0.11–0.41. No criterion applies to them: `λ_m` is the median point of the symmetric law
  (`β = 0`), so the ratio is not the right yardstick when `β^te ≠ 0`. Reported, not judged.
- The training-size failures of run 1 (P1, P4) are therefore not a property of the limits: with one
  lower column and equal class scales the rate, the law of `H` and the coin all appear at
  `n ≤ 5·10⁵`.
