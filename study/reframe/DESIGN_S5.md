# Supplement S5 — the limit law of Cor. `cor:detwindow`, and a one-sided `(x, |x|³)` check in which the collapse happens

Written 2026-11-24, after run 4. It answers two questions raised in review of the manuscript:
1. **Item 1.** Check the limit law of Cor. `cor:detwindow` itself. D3 of run 4 checked only the
   finite-`n` step of its proof.
2. **Item 2.** A non-vacuous one-sided check of `(x, |x|³)`. A2 of run 4 passed only because its
   fits did not collapse: `β^te = +1` and `K̂_t > 0` in most fits.

The thresholds below are registered here and are not moved after the run. Post-run notes go to a
dated section at the end. `DESIGN*.md` of runs 1–4 are not touched.

**What existed before registration:**
- the design probe (`design_probe_s5.py` → `results/design_probe_s5.json`). It ran the training fits
  of block G. For block F it computed one fitted vector per dictionary at the largest `n`;
- a check of the exact AUC against the conditional limit for three fits per dictionary;
- a smoke test of the controls, and of one G cell (16 fits) and one F cell (16 fits).
  - The G smoke test showed one thing used below: at `ε^te = 0.4` the AUC peak of a collapsing
    fit can stay below 0.75. The onset `N_on` (first `N` past the peak with `AUC ≤ 0.75`) is then
    undefined, so G1 is stated through the drop from the peak and not through `N_on`.
  - No other window output was read.

## A defect found while designing: the LePage sum of runs 2–4 drops its tail

`h1v4_s2.lepage` draws `Z_{γ,β} = Σ_{k ≤ 4000} ξ_k Γ_k^{−1/γ}` and adds nothing for `k > 4000`.
- **Size.** For `β ≠ 0` the missing tail has mean `β γ/(1−γ) Γ_K^{1−1/γ}`:
  - about `0.19 β` at `γ = 0.75`;
  - about `0.13 β` at `γ = 0.73`;
  - about `0.95 β` at `γ = 0.83`.
  - It is below `3·10⁻⁴` for `γ ≤ 0.5`, which covers every one-sided cell of runs 2 and 3, and
    B4, C4 and E4 of run 4.
- **Where it matters.** Only in the one-sided A4 cells of run 4 (`γ = 0.73, 0.83`). There the
  fits with `K̂_t > 0` predict exactly 1 either way; the remaining fits get a biased prediction.
- **Check.** Against the Gil–Pelaez CDF (`h1v4_s5.stable_cdf`), the uncompensated sum is off by
  0.012–0.17 at `β = ±1`. The compensated sum `h1v4_s5.lepage_c` adds the tail mean and a normal
  term with the tail variance, and is off by ≤ 0.001.
- **In S5.** Every S5 prediction uses `lepage_c`. The prediction of `F.summarize` (uncompensated)
  is kept as `predErr_uncomp` for the record. `criteria_s5.py` re-predicts the run-4 one-sided A4
  cells from their stored per-fit values (reported, not a criterion).

## Blocks

**S5-F (the limit law of Cor. `cor:detwindow`).** The design of S3-C:
- Dictionaries:
  - `oo4 = (sgn(x)|x|, sgn(x)|x|⁴)`: same parity, sign locked;
  - `oe4 = (sgn(x)|x|, |x|⁴)`: opposite parity, sign a coin.
- `α_c = 3`, `a_L = 1`, `a* = 4`, `γ = 3/4`.
- Clean laws `N(0.3, 1)` and `N(0, 1)`, training split `(0.05, 0.05)`.
- The fits are the S3-C fits: same seeds, 300 per `n`, at `n ∈ {32 000, 128 000, 512 000}`. S3-C
  measured the Kolmogorov distance of `H_n` to `H` at `n = 512 000`: 0.068 (`oo4`), 0.041 (`oe4`).
- **Window length.** Fixed in advance: `N_n = ⌈κ₀ (c₀ n ε^tr)^{(a*−a_L)/(a*−α_c)}⌉ = ⌈κ₀ (0.1 n)³⌉`.
- **Registered rule for `κ₀`** (`h1v4_s5.kappa0_rule`): `κ̃ |K^sub_{l*}| median|H| = λ_m`, so the
  median fit of the limit law sits at AUC 0.75 under the symmetric split. It gives:
  - `κ₀ = 50.39` (`oo4`) and `812.4` (`oe4`);
  - `N_n` from `1.7·10¹²` to `1.1·10¹⁷`.
- **Test splits.**
  - `(0.05, 0.05)`: `β^te = 0`;
  - `(0.1, 0)`: `β^te = 0` for `oo4` and `+1` for `oe4`. In `oe4` the fits with `K̂_t > 0` then
    have limit 1 and the others collapse, so the limit law is two-sided.
- **The AUC is exact, with no window Monte Carlo.** `AUC_{N_n}(K̂)` comes from the
  characteristic-function engine of run 1 (`CFEngine`, Gil–Pelaez), with the clean laws passed in
  (`CFEngineClean`).
  - The limit law `Ψ(−κ̃ K^sub_{l*} H)` uses 20 000 draws of `H` (`h_sample`, `campbell=True`) and
    200 000 draws of `Z_{γ,β^te}` (`lepage_c`).
- **Distance.** The two laws are compared by the Kolmogorov distance read on the grid
  `u ∈ {0.01, …, 0.99}`: `max_u |F_fit(u) − F_lim(u)|`.
  - The grid is needed because the `oe4` one-sided limit has an atom at 1. A fit at `1 − 10⁻⁹`
    must not count as a miss.
- **Two contributions to the distance, reported per cell:**
  - the window step: fit AUC against the conditional limit `Ψ(x_n)`,
    `x_n = (N_n/N*_n)^{(a*−α_c)/α_c} sgn K̂_t`;
  - the training step: `Ψ(x_n)` against the limit law.

**S5-G (`(x, |x|³)`, one-sided splits that make the fits collapse).**
- Dictionary `oe3 = (x, |x|³)`: `a_L = 1`, `a* = 3`, `α_c ∈ (2, 3)`; `R = 200` fits per cell.
  Assumptions (B4)–(B5) hold, as in A4 of run 4.
- A fit collapses past `N*` exactly when `sgn(K̂_t) β^te = −1`. These are the "collapsing fits".
  - In run 4 nearly every fit had `K̂_t > 0` and the splits were forward (`β^te = +1`), so almost
    none collapsed.
- **Clean laws.**
  - `gauss`: those of runs 1–4, `N(0.3, 1.3²)` and `N(0, 1)`;
  - `mirror`: `N(0.15, 1)` and `N(−0.15, 1)`. Every even moment is equal, so the clean even
    difference vanishes and the sign of `K̂_t` is close to a coin.
- **Splits.** Forward `(ε, 0)` gives `β^te = +1`; reverse `(0, ε)` gives `β^te = −1`.
- **Registration rule** (from the probe; 40 fits; candidates `α_c ∈ {2.2, 2.5}`, `n ∈ {200, 1000}`,
  `ε ∈ {0.1, 0.2, 0.4}`, both clean laws, both directions). A cell enters G1–G3 when:
  - the share of collapsing fits is ≥ 0.25;
  - the share with `μ̂ ≤ 0` is ≤ 0.10;
  - over the collapsing fits, median `N*` and median `N*_eff` both lie in `[50, 10⁴]`, so the
    collapse falls inside the grid (`N_max = 31 623`).
  - `N*_eff` is `N*` with the admissible drift `μ̂` replaced by `μ̂ + K̂_t Δ_clean`. Here
    `Δ_clean = (1−ε^te_+) E_{G+}|x|³ − (1−ε^te_−) E_{G−}|x|³` is the clean drift of the top
    column, which the theory drops as `n → ∞` (`diag_s4_clean_drift.py` of run 4). This term is
    never zero under a one-sided split, even with mirror clean laws.
- **Cells.** Three cells pass the rule, all at `α_c = 2.2`, `n = 200`:

  | clean | split | collapsing share | median `N*` | median `N*_eff` |
  |---|---|---|---|---|
  | gauss | `(0, 0.4)` | 0.93 | 185 | 2 250 |
  | mirror | `(0.4, 0)` | 0.65 | 6 050 | 7 810 |
  | mirror | `(0, 0.4)` | 0.35 | 2 630 | 3 680 |

  - At `α_c = 2.5` no candidate does: `N*_eff ≥ 3.8·10⁵`.
- **Two more "drift" cells**, registered for G3 only. In each, `N*` and `N*_eff` differ by about
  10×, so the two predictions disagree:
  - gauss `(0, 0.2)`, `n = 200`: `N*` 2 490, `N*_eff` 24 900;
  - gauss `(0, 0.4)`, `n = 1000`: `N*` 6 700, `N*_eff` 16 900.

## Controls (must pass before the grid; a miss aborts the run)

- **C16.** The S5 fits reproduce the stored S3-C median `|K̂_t|` at all three `n`, for both
  dictionaries (relative difference ≤ 1e-12).
- **C17.** `CFEngineClean` with the run-1 clean laws equals `H.CFEngine` (`AUC_N`,
  `N ∈ {100, 10⁶}`; difference ≤ 1e-12).
- **C18.** `lepage_c` matches the Gil–Pelaez CDF of `Z_{γ,±1}` at `γ ∈ {0.733, 0.75}` and 6 points
  each (difference ≤ 0.003; the SE of the empirical CDF from 200 000 draws is ≤ 0.0012).

## Registered criteria

- **F1 — the limit law.**
  - Requirement: at `n = 512 000`, in all four F cells (`oo4`, `oe4` × symmetric, one-sided),
    grid-Kolmogorov distance between `AUC_{N_n}(K̂)` over the 300 fits and the law of
    `Ψ(−κ̃ K^sub_{l*} H)` ≤ 0.10.
  - Basis: the 5 % critical value for 300 against 20 000 draws is 0.079, and S3-C measured
    `H_n` against `H` at 0.068 and 0.041 at this `n`.
- **F2 — the mean.** At `n = 512 000`, in all four cells, `|mean AUC_{N_n} − E Ψ| ≤ 0.03`.
- **F3 — the window step.** In all 12 F cells, the median over fits of `|AUC_{N_n} − Ψ(x_n)|`
  ≤ 0.01.
- **G1 — the collapse happens.** In the three collapse cells, over the collapsing fits:
  - median `AUC_{N_max}` < 0.5;
  - median drop from the peak, `peak − AUC_{N_max}`, ≥ 0.25.
- **G2 — the limit of Thm `thm:crossover` past the crossover.** In the three collapse cells, the
  median over collapsing fits of `|AUC_{N_max} − pred|` ≤ 0.10, with `pred` from `N*`.
- **G3 — the clean-drift prediction, registered.** In all five G cells, the median over collapsing
  fits of `|AUC_{N_max} − pred_eff|` ≤ 0.10, with `pred_eff` from `N*_eff`. In run 4 this
  prediction was post-hoc; here it is stated before the windows.
- **Reported, not criteria:**
  - the F distances at `n = 32 000` and `128 000` (the trajectory), and the window and training
    parts;
  - per cell: lock, `P̂{K̂_t > 0}`, bootstrap 95 % intervals of the medians and means;
  - in the G cells: the winsorized `AUC_{N_max}` of the collapsing fits; the median `AUC_{N_max}`
    of the non-collapsing fits; the uncompensated prediction error;
  - the run-4 one-sided A4 cells re-predicted with `lepage_c`.

A criterion that fails is a finding, recorded with its cell, not a reason to change the grid.

## Post-run notes (2026-11-24)

Run `results/h1v4-s5-2026-11-24/`: 11 cells, 10 302 s, no failed fit.
- Controls: C16 relative difference 0; C17 difference 0; C18 max difference 0.0009.
- Verdicts by `criteria_s5.py` (`results/criteria_s5.json`):

| criterion | verdict | detail |
|---|---|---|
| F1 limit law | PASS 4/4 | grid-Kolmogorov distance at `n = 512 000`: 0.065 / 0.061 (`oo4` sym / one), 0.035 / 0.040 (`oe4`); window part ≤ 0.010, training part 0.040–0.065 |
| F2 mean | PASS 4/4 | `|mean AUC − E Ψ|` ≤ 0.014 at `n = 512 000` (0.035 at `oe4` one, `n = 32 000`) |
| F3 window step | PASS 12/12 | median per-fit `|AUC_{N_n} − Ψ(x_n)|` ≤ 0.0011 at `N_n` from 1.7·10¹² to 1.1·10¹⁷ |
| G1 collapse | PASS 3/3 | collapsing fits 43–88 %; median peak 0.58–0.60, median `AUC_{N_max}` 0.000, drop 0.55–0.59 |
| G2 limit with `N*` | PASS 3/3 | median error 0.000 |
| G3 limit with `N*_eff` | PASS 5/5 | median error 0.000 |

- **The training step dominates the distance in F.** `H_n` against `H` gives 0.040–0.070, as S3-C
  measured; the window step adds ≤ 0.010.
  - The `oe4` one-sided cells have a two-sided limit law, with an atom at 1 for `K̂_t > 0`. It is
    reproduced as well.
- **G2 and G3 do not discriminate.** Past the crossover every collapsing fit has
  `AUC_{N_max} = 0`. Both predictions saturate at 0, so the error is 0 by construction, as in block
  E of run 4.
  - A post-hoc comparison, not registered, separates them: the median `AUC_N` curve (`N ≥ 100`)
    against the median predicted curve. Maximum deviation with `N*` against `N*_eff`:
    - gauss `(0, 0.4)`, `n = 200`: 0.36 against 0.08;
    - gauss `(0, 0.2)`, `n = 200`: 0.49 against 0.06;
    - gauss `(0, 0.4)`, `n = 1000`: 0.11 against 0.09;
    - mirror cells, where the clean drift is small: 0.08–0.10 with both.
  - The clean-drift term of run 4 therefore also sets where the one-sided collapse happens.
- The winsorized detector stays at 1.000 in the collapsing fits of every G cell.
- **Run-4 one-sided A4 cells, re-predicted with `lepage_c`.**
  - No per-fit prediction moves by more than 0.027, and the median errors stay 0.000.
  - A2 keeps its verdict, and stays vacuous.
