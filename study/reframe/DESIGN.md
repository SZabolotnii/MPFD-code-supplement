# Stage 6 — Monte Carlo design for hypothesis H1 v4 (case |I| = 1)

Written 2026-11-22, before any run. The thresholds below are registered here and are not
moved after the run; post-run notes go to a dated section at the end.

## Model

Tail index `α_c = 3`. Contaminant `Q`: signed Pareto, `|C| = U^{-1/3}` (`U ~ Unif(0,1]`), so
`P(|C| > x) = x^{-3}` for `x ≥ 1` (`c_0 = 1` exactly); `P(C > 0) = p_+`. Clean laws
`G_+ = N(0.3, 1.3²)`, `G_- = N(0, 1)` (both odd and even columns carry signal).

Dictionaries (parity, exponent), top column `t` last, top admissible `l*` second:

| name | columns | parity of (l*, t) |
|---|---|---|
| `oo` | `|x|^0.5`, `x`, `sgn(x)|x|^a*` | same (odd, odd) |
| `oe` | `|x|^0.5`, `x`, `|x|^a*` | opposite |
| `ee` | `sgn(x)|x|^0.5`, `|x|^1.2`, `|x|^a*` | same (even, even) |
| `eo` | `sgn(x)|x|^0.5`, `|x|^1.2`, `sgn(x)|x|^a*` | opposite |

`a* ∈ {2.4, 4}`: `2.4` is regime (R) only (`α_c/2 < a* < α_c`), `4` is (R) and (B). All of
(B4) `a* + a_L > 3` and (B5) (no `a_j`, `2a_j`, `a_i + a_j` equal to 3) hold for every
dictionary; `2a_l < 3` for the admissible columns.

Fitted detector exactly as in `paper/reframe/main.tex`: per-class sample covariance with
divisor `n`, `F̂ = Σ̂_+ + Σ̂_-`, `Ŷ = m̂_+ − m̂_-`, `K̂ = F̂^{-1}Ŷ` (solved after diagonal
equilibration; `cond` of the equilibrated `F̂` reported). `K^sub` and `μ^te_L` are population
values (Gaussian quadrature + closed-form Pareto moments `E|C|^q = 3/(3−q)`).

## Block A — training contamination (MC), all four dictionaries × `a*` × `p_+ ∈ {0.5, 0.8}`
× training split `(ε^tr_+, ε^tr_-) ∈ {(0.05, 0.05), (0.10, 0), (0, 0.10)}` (48 cells)

`n ∈ {500, 2000, 8000, 32000, 128000}` per class, `R = 300` training replications per `n`.
Per `n`: median `|K̂_t|`; sign-lock frequency `P̂{sgn K̂_t = −sgn K^sub_{l*}}`; `P̂{K̂_t > 0}`;
median `‖K̂_L − K^sub‖`; median `|AUC_10(K̂) − AUC_10((K^sub, 0))|` on clean test windows
(`N = 10`, 20 000 paired windows, common to both vectors). At the largest `n`: the KS
distance between `−b_n^{a*−a_L} K̂_t / K^sub_{l*}` and a reference sample of `H` (20 000
draws of the Poisson series, 5 000 terms plus the remainder correction).

## Block B — contaminated test windows (exact CF inversion), `a* = 4`

Training: `ε^tr = (0.05, 0.05)`, `p_+ = 0.5`, `n ∈ {50, 200, 1000}`, 40 replications per
`n` and dictionary (12 cells). For each fitted `K̂`, the conditional `AUC_N(K̂)` is computed
by Gil-Pelaez inversion of the exact characteristic function of one window pair
(quadrature for the Gaussian part; panel quadrature in `log x` with an integration-by-parts
tail for the Pareto part), for `N = 10^1 … 10^16` (step `10^0.05`).
- S1, `ε^te = (0.05, 0.05)`, symmetric contaminant (`β^te = 0`, limit ½): collapse onset
  `N_on` = first `N` past the peak where `AUC_N ≤ 0.75`; compared with the theorem's
  `N*_n(K̂)` through the ratio `N_on / (N*_n λ_m^{3})`, `λ_m` the root of
  `P(1 + λ Z_{3/4,0} > 0) = 0.75` (expected → 1).
- S2, `ε^te = (0.10, 0)`: `AUC_N` at `N = 10^6 N*_n` against `ρ(3/4, sgn(K̂_t) β^te)`
  (`β^te = 1` for an even top, `0` for an odd top with `p_+ = ½`).

## Controls (must pass before the grid; a miss aborts the run)

C1 CF engine, Gaussian only: `AUC_20` of the linear score equals the normal formula (tol
1e-6). C2 CF engine vs brute-force MC at `N = 50` and `N = 2000` for a contaminated window
(tol 4 MC standard errors). C3 stable quantile from the CF engine vs `scipy.stats.levy_stable`
(tol 1e-3). C4 `H` sampler: `P(H > 0) = 1` (same parity) and `0.5 ± 0.015` (opposite,
symmetric). C5 moment engine: population means of the admissible columns under the
training mixture vs a 4·10⁶-draw sample (tol 4 SE).

## Registered criteria (hypothesis H1 v4 falsifiers, made numeric)

- P1 (R slope). In every Block A cell, the log-log slope of median `|K̂_t|` over the three
  largest `n` is within 0.10 of `−(a* − a_L)/α_c`.
- P2 (R AUC). In every Block A cell, median `|ΔAUC_10|` at `n = 128000` is ≤ 0.01 and below
  its value at `n = 500`.
- P3 (sign-lock). In every same-parity cell (all splits, both `p_+`), the lock frequency at
  `n = 128000` is ≥ 0.95.
- P4 (fair coin). In every opposite-parity cell with `p_+ = 0.5`, `|P̂{K̂_t > 0} − ½| ≤ 0.087`
  (3 SE at `R = 300`) at `n = 128000`.
- P6 (B onset slope). The log-log slope of median `N_on` over `n ∈ {50, 200, 1000}` is within
  0.15 of `(a* − a_L)/(a* − α_c)` (3.0 for `oo`, `oe`; 2.8 for `ee`, `eo`).
- Reported, not criteria: KS distance to `H` (P5), the `N_on/N*` ratio (P7), the S2 limit
  error (P8), `cond(F̂)`.

A criterion that fails is a finding for the author, recorded with its cell, not a reason to
change the grid.

## Post-run notes (2026-11-22)

Run `results/h1v4-2026-11-22/` (controls C1–C5 passed; 52 cells; `criteria.py` reproduces every
number below from the hashed artifacts).

**Deviation from the design, Block B.** For some fitted vectors the window CF decays only
polynomially at large `t` (the `|x|^0.5` kink at 0), so `|ψ_D|^N` had not decayed by the end of
the `t`-grid for `N ≈ 10–14`. Extending the grid was infeasible (Pareto panel quadrature at
large `t`); those `AUC_N` are `nan` and counted. The first finite `N` is ≤ 14.1 in every
replication, and the median peak is at `N = 158–238`, so neither the peak nor `N_on` is
affected. S2 points with `10^6 N* > 10^18` are `nan` (3–25 % per `n`).

| | verdict | cells |
|---|---|---|
| P1 slope of median `|K̂_t|` within 0.10 | **FAIL** | 13/48 pass; `a* = 2.4`: errors +0.18…+0.33 (same parity) and −0.13…−0.44 (`oe`); `a* = 4`: 11 of 24 pass, errors up to ±0.22 |
| P2 `ΔAUC_10` → 0 | PASS | 48/48 (≤ 0.0027 at `n = 128000`) |
| P3 sign-lock, same parity | PASS | 24/24 (lock ≥ 0.99 at `n = 128000`, every split and `p_+`) |
| P4 fair coin, opposite parity, `p_+ = ½` | **FAIL** | 6/12; `oe`: `P̂{K̂_t > 0}` = 0.79–0.89 (`a* = 2.4`), 0.10–0.45 (`a* = 4`); `eo minus 2.4`: 0.61 |
| P6 onset slope in `n` within 0.15 | **FAIL** | 0/4; slopes 1.15 (`oe`), 1.93 (`eo`), 1.94 (`ee`), 2.35 (`oo`) vs 3.0 / 2.8 |
| P7 `N_on / (N*_n(K̂) λ_m³)` (reported) | — | at `n = 1000`: 0.997 (`oo`), 1.018 (`eo`), 0.960 (`ee`), 1.361 (`oe`) |
| P8 S2 limit `ρ(3/4, sgn(K̂_t) β^te)` (reported) | — | median error ≤ 0.0028 in every cell |
| P5 KS to `H` at `n = 128000` (reported) | — | `a* = 4`, same parity: 0.055–0.20; `a* = 2.4`: 0.23–0.52 |

**Reading (post hoc, not registered).** The failures are pre-asymptotic, and two diagnostics
locate the terms the proofs discard as `o(·)`:
- D1 (`diag_competitor.py`): removing the lower even column of `oe` moves `P̂{K̂_t > 0}` from
  0.21 → 0.66 (`a* = 4`, bal) and 0.87 → 0.98 (`a* = 2.4`), i.e. towards 1, not ½ — the lower
  same-parity column is one of two biases, not the cause.
- D2 (`diag_clean_mean.py`): with equal clean scales (`G_+ = N(0.3, 1)`) the reduced `oe` gives
  0.507 (`a* = 4`, bal; fair), but 0.78 / 0.31 under the one-sided splits (0.94 / 0.08 for
  `a* = 2.4`). The two terms are the clean part of `Ŷ_t` (`E_{G+}|x|^{a*} − E_{G−}|x|^{a*}`,
  6.49 at `a* = 4` with the registered laws) and its contamination part `±b_n^{a*}/n`, whose
  ratio to the cross term is `b_n^{-a_L}/|K^sub_{l*}|` — about 0.5 at `n = 128000` with
  `K^sub_{l*} ≈ 0.08`. At finite `n` the opposite-parity sign therefore follows the training
  imbalance and the class scale difference; the fair coin is an `n → ∞` statement.
- P6 fails for the same reason one level up: at `n ≤ 1000` the same-parity lock is still
  ≈ 0.5 (in Block A the smallest lock frequency over the 24 same-parity cells is 0.50 at
`n = 2000`, 0.85 at `n = 32000` and 0.99 only at `n = 128000`), so `|K̂_t|` is not yet in its limit law.
  Given `K̂`, the crossover formula itself is accurate (P7 ≈ 1 for three of four dictionaries).
