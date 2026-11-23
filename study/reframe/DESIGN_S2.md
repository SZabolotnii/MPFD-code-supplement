# Stage 6, supplement S2 — practical window lengths, a second contaminant, a winsorized comparator, real clean laws

Written 2026-11-22, after the run `results/h1v4-2026-11-22/` and before any S2 test window
exists. The author asked for this supplement after the paper-plan review
(internal planning note, not included).
That review objected that in the registered design the collapse sets in only at
`N ≈ 10⁷–10¹⁰`. The thresholds below are registered here and are not moved after the run.
Post-run notes go to a dated section at the end. The first run and its notes in `DESIGN.md` are not
touched.

## What is new, and what is reused

- The detector, the fit (`h1v4.fit`), the dictionaries `oo, oe, ee, eo` and their exponent
  layout are all reused, as are the synthetic clean laws `G₊ = N(0.3, 1.3²)` and
  `G₋ = N(0, 1)` and the Pareto contaminant (`α_c = 3`, `c₀ = 1`). Only the top exponent
  changes: `a* = 8` instead of 4. All of (B4) and (B5) hold: the exponents are 0.5, 1 or 1.2,
  and 8; no sum equals 3.
- **Training contamination:** `ε^tr = (0.05, 0.05)`, `p₊ = ½`, as in Block B.
- **Test contamination:** total `ε^te = 0.4`, either symmetric `(0.2, 0.2)` or one-sided
  `(0.4, 0)`. One-sided cells are run for the even tops only (`oe`, `ee`). For an odd top with
  `p₊ = ½`, `β^te = 0`, so a one-sided cell is the same experiment as a symmetric one.
- **Window AUC.** It is measured by Monte Carlo on windows, not by CF inversion. At `a* = 8`
  the panel quadrature of the Pareto part needs more than 2·10⁶ panels at `t ≈ 166` (feasibility
  probe, 2026-11-22). At `N ≤ 3·10⁴` direct simulation is cheap.
  - For each replication there are `M = 500` independent window pairs for each class.
  - Each pair has length `N_max = 10^{4.5} ≈ 31 623`.
  - `AUC_N` is read from prefix sums at `N ∈ 10^{0.5, 0.625, …, 4.5}`, rounded to integers
    (33 points).
  - On the **same** windows three score vectors are evaluated: unclipped `K̂`, winsorized
    `K̂^W`, and the fitted sub-dictionary `K̂^sub` (top column dropped, refitted).
  - MC standard error of one `AUC_N` ≤ 0.022.
- **Cell selection.** Cells were selected by a design-time probe that uses training fits only
  (`design_probe_s2.py` → `results/design_probe_s2.json`). At `a* = 8` and `ε^te = 0.4` the
  median of `N*_n(K̂)` over 40 fits is 321–428 at `n = 50`, 822–1568 at `n = 200` and
  2804–5396 at `n = 1000`, for all four dictionaries. With the onset factor
  `λ_m^{α_c/(a*−α_c)} = 0.570` (`γ = 3/8`), the predicted `N_on` lies in roughly `10²–7·10³`,
  inside the grid. No AUC was computed by the probe.

## Blocks

**S2-A (Pareto, synthetic clean laws).**
- Dictionaries `oo, oe, ee, eo` × `n ∈ {50, 200, 1000}` per class × symmetric test
  contamination: 12 cells.
- Plus `oe, ee` × `n` × one-sided test contamination: 6 cells.
- `R = 40` training replications per cell.

**S2-B (Student-t₃ contaminant).**
- `Q = t₃`: tail index 3, symmetric, `c₀ = 2·3·Γ(2)/(√(3π)·Γ(3/2)) = 2.2053` (the `t_ν` density is `~ Γ((ν+1)/2) ν^{(ν+1)/2} / (√(νπ) Γ(ν/2)) · x^{−(ν+1)}`) (closed
  form; checked by control C7).
- Training block, as Block A of the first run but with `Q = t₃`:
  - `ee` and `oe`, `a* = 4`, split `(0.05, 0.05)`, `p₊ = ½`;
  - `n ∈ {2000, 32000, 128000}`, `R = 100`;
  - measured: lock frequency, `P̂{K̂_t > 0}`, and median `|ΔAUC₁₀|` against the population
    sub-dictionary (clean test windows, `N = 10`, 20 000 pairs).
- Crossover cells: `ee, oe`, `a* = 8`, `n = 200`, symmetric and one-sided, `R = 40`
  (4 cells, window MC as in S2-A).
- `K^sub` and the test means use the closed-form `t₃` moments
  `E|T|^q = 3^{q/2} Γ((q+1)/2) Γ((3−q)/2) / (√π Γ(3/2))` (odd moments 0).

**S2-C (winsorized comparator).** This block has no cells of its own; it is evaluated in
every S2-A, S2-B and S2-D crossover cell.
- Thresholds are the empirical 0.5 % and 99.5 % quantiles of the **pooled training sample**
  (both classes, contamination included).
- `x` is clipped to these thresholds before `φ`, and `K̂^W` is refitted on the clipped training
  features. Test windows are clipped with the same training thresholds.
- Nothing about the contaminant is assumed known.

**S2-D (semi-synthetic, real clean laws).**
- Source: RflyMAD real flights without faults (lake slug `rflymad-real-nofault`, used through
  the vertical-acceleration series cached in the file named by `MPFD_RFLY_CACHE` (see README)).
- Airborne, steady-altitude samples only: height above the flight's minimum `> 3 m` and
  `|v_z| < 0.5 m/s`.
- Each flight's acceleration is centred by its airborne median.
- Classes:
  - `G₋` = hover (`v_xy < 1 m/s`);
  - `G₊` = cruise (`v_xy > 4 m/s`).
- Flights with at least 500 samples in each class are eligible; there are 29 (probe 2026-11-22).
- The eligible flights are split once, by a seeded permutation, into a training half and a test
  half (the flight is the dependence unit).
- Data are standardised by the pooled sd of the training half.
- Training and test samples are **i.i.d. draws from the per-class pools** of their half, so A1
  holds by construction and (C1) holds because a pool is bounded.
- Probe values (all flights pooled): hover sd 0.64, Hill(200) 7.5; cruise sd 1.47, Hill(200) 5.0.
  The classes differ almost only in scale, an even difference.
- Grid: `ee, oe`, `a* = 8`, `n ∈ {200, 1000}` × symmetric and one-sided Pareto test
  contamination, `R = 40` (8 cells). `K^sub`, `μ` and `N*` use the empirical pool moments plus
  the closed-form Pareto moments.
- The dependence inside a real window is not represented. This is said in the paper, and no
  claim about contiguous windows is made.

## Controls (must pass before the grid; a miss aborts the run)

- **C6, window MC vs exact CF.** The window-MC engine is checked against `CFEngine` at
  `a* = 4` (where CF works): one `ee` fit at `n = 200`, symmetric `ε^te = (0.05, 0.05)`,
  `N ∈ {100, 1000}`, `M = 4000` pairs. Tolerance: 4 MC standard errors.
- **C7, the t₃ sampler and its constant.**
  - The empirical `P(|T| > x)` from 4·10⁶ draws at `x ∈ {10, 20}` must match the exact `t₃`
    tail (`scipy.stats.t`) within 4 SE.
  - The exact `x³ P(|T| > x)` at `x = 10³` must lie within 0.1 % of `c₀ = 2.2053`.
  - `E|T|^q` for `q ∈ {0.5, 1.2, 2}` must match the sample within 4 SE.
- **C8, RflyMAD pools.**
  - 29 eligible flights.
  - The training/test split is a function of the seed only.
  - Pools are non-empty in all four (half, class) combinations.
  - The class sd ratio of the training half lies within `[1.5, 3]`.
- **C9, winsorizer.** Thresholds depend on the training sample only: recomputing them after a
  change of the test windows leaves them unchanged. `K̂^W` is finite.
- **C10, the conditional prediction.** It comes from the crossover theorem,
  `P{1 + c^{(a*−α_c)/α_c} s Z_{γ,β} > 0}` with `c = N/N*_n(K̂)` and `s = sgn K̂_t`.
  - It is evaluated by LePage series sampling of `Z_{γ,β}` (2·10⁵ draws, 4000 terms).
  - For `β = 0` it must agree with `h1v4.stable_cdf_sym` to 0.005 at three points.

## Registered criteria

- **Q1 — collapse at practical `N`.** Applies to every symmetric cell of S2-A, S2-B and S2-D.
  - The median over replications of `max_N AUC_N(K̂)` is ≥ 0.80.
  - `N_on` (first `N` past the peak with `AUC_N ≤ 0.75`) is finite in ≥ 80 % of replications.
  - The median `N_on` is ≤ 2·10⁴.
- **Q2 — onset formula at practical `N`.** In every symmetric cell of S2-A and S2-B, the median
  over replications of `N_on / (λ_m^{α_c/(a*−α_c)} N*_n(K̂))` lies in `[0.67, 1.5]`. `N*` uses
  `c₀ = 2.2053` for `t₃`. S2-D is reported, not judged: the pools are not Gaussian and the
  theorem needs only (C1), but its `μ` comes from finite pools.
- **Q3 — limit past the crossover, one-sided test contamination.** In every one-sided cell of
  S2-A, S2-B and S2-D, the median over replications of `|AUC_{N_max}(K̂) − pred|` is ≤ 0.10.
  - `pred` is the C10 conditional prediction at `c = N_max / N*_n(K̂)`.
  - For an even top, `β^te = 1`.
- **Q4 — the training limits survive a non-Pareto contaminant (S2-B training block).**
  - `ee`: lock frequency ≥ 0.95 at `n = 128000`.
  - `ee` and `oe`: median `|ΔAUC₁₀|` ≤ 0.01 at `n = 128000`.
- **Q5 — winsorizing.** A question for the reader, with either answer reported:
  - (a) in every symmetric crossover cell (S2-A, S2-B, S2-D), the median `AUC_{N_max}(K̂^W)`
    is ≥ 0.90 — clipping removes the collapse;
  - (b) in the same cells, the median of `AUC_{10}(K̂^W) − AUC_{10}(K̂^sub)` on **clean** test
    windows is ≥ −0.02 — clipping costs at most 0.02 AUC against the sub-dictionary at short
    windows.
- **Reported, not criteria:**
  - lock frequency and `P̂{K̂_t > 0}` at the small `n` of the crossover cells (the first run
    showed they are pre-asymptotic there);
  - `cond(F̂)`;
  - the share of replications with `N_on = nan`;
  - the AUC curves of `K̂^sub`.

A criterion that fails is a finding for the author, recorded with its cell, not a reason to
change the grid. Failed MC replications (a non-finite `K̂`) are counted per cell, never dropped
silently.

## Post-run notes (2026-11-22)

**Run.** `results/h1v4-s2-2026-11-22/`: 32 cells plus 5 controls, `MANIFEST.sha256`, `RUN.json`,
`REPORT.md`; 976 s. `criteria_s2.py` reproduces every number below from the hashed artifacts.
No replication failed (`failed = 0` in every cell).

**Controls.** All passed:

| Control | Result |
|---|---|
| C6 (MC vs CF) | z = 0.09 and −0.003 |
| C7 | tail z ≤ 2.5; exact `x³P(|T|>x)` at `10³` is 2.20531 (relative error 4·10⁻⁶); moments z ≤ 0.9 |
| C8 | 29 flights; split repeats; sd ratio 2.38 |
| C9 | passed |
| C10 | maximum error 0.0021 |

**Verdicts.**

| | verdict | cells |
|---|---|---|
| Q1 collapse at practical `N` (symmetric cells) | **FAIL** | 14/18. Median peak ≥ 0.80 in all 18 (0.808–0.992); median `N_on` 238–16 283. Failures: `N_on` finite in < 80 % of replications in `oo`, `ee`, `eo` at `n = 50` (0.60, 0.72, 0.75) and in RflyMAD `oe`, `n = 1000` (0.55; median `N*` = 5.4·10⁴, beyond `N_max`) |
| Q2 onset formula (symmetric, synthetic + t₃) | PASS | 14/14; `N_on / (0.570 N*(K̂))` = 0.95–1.13 |
| Q3 limit past the crossover (one-sided) | PASS | 12/12; median `|AUC_{N_max} − pred|` ≤ 0.017 |
| Q4 training limits with t₃ | PASS | 3/3; `ee` lock 0.97 / 1.00 / 1.00 at `n = 2000 / 32000 / 128000`; median `|ΔAUC₁₀|` ≤ 0.0017 |
| Q5a winsorizing removes the collapse | PASS | 18/18; median `AUC_{N_max}(K̂^W)` = 1.000 in every symmetric cell |
| Q5b winsorizing costs ≤ 0.02 at `N = 10` | PASS | 18/18; median `AUC₁₀(K̂^W) − AUC₁₀(K̂^sub)` within [−0.007, +0.009] |

**Reported, not criteria.**
- **Same-parity lock at the small `n` of the crossover cells.**
  - Synthetic laws: 0.40–0.65 at `n ≤ 200`, and 0.88–0.90 for `ee` at `n = 1000`.
  - RflyMAD clean laws: `ee` 1.00 at `n = 200` and `n = 1000`.
- **One-sided cells.** The end AUC follows the sign of each `K̂_t`, as the theorem says
  conditionally. On synthetic laws at small `n` the sign is close to a coin flip, so the median end
  AUC is 1.000 in five of six cells. On RflyMAD `ee` the sign is locked, and the window score
  inverts reproducibly: median end AUC is 0.006 at `n = 200`, and 0.251 at `n = 1000`, where `N_max`
  is still before the end of the transition.
- **`N_on / (0.570 N*)` in one-sided cells is 0.19–0.23.** The symmetric `λ_m` does not apply
  under `β = 1`, so this ratio is not a criterion.
- **t₃ `oe`.** `P̂{K̂_t > 0}` = 0.27 at `n = 128000`, not ½: the same pre-asymptotic bias as the
  Pareto `oe` in the first run (FINDINGS 2026-11-22 SCOPE).
- **The fitted sub-dictionary** also reaches AUC 1.000 at `N_max` in every symmetric cell.

**Reading (post hoc, not registered).**
- **The collapse is practical.** With a top exponent far above the tail index (`a* = 8`,
  `α_c = 3`) and heavy test contamination (`ε^te = 0.4`), the AUC of the contaminated-trained
  detector peaks at 0.81–0.99 and then falls to ½ within `N ≈ 10²–10⁴`.
- **The onset formula holds where it is defined.** It holds to within 13 % at
  `n ∈ {50, 200, 1000}`, for a non-Pareto contaminant, and on real clean laws.
- **Q1 fails only on its "finite in 80 %" clause.** At `n = 50` a quarter to two fifths of the
  fits do not cross 0.75 inside the grid. For the real-data `oe` cell at `n = 1000`, `N*` lies past
  `N_max`. The selection probe used the Gaussian laws, so it did not cover this cell.
- **Winsorizing at the training-sample 0.5 / 99.5 % quantiles removes the collapse in every
  cell, at no measurable cost on clean short windows.** So does dropping the top column. This bears
  on kill criterion K3 (written for the fixed-`K` frame of H1 v1: "clipping within 0.01 AUC
  of the best unclipped dictionary in every cell … then a sizing theory for unclipped detectors has
  no use case"). Whether K3 fires for the H1 v4 paper is the author's decision.
