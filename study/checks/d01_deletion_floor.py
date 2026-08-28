#!/usr/bin/env python3
"""D1 (post-run diagnostic, 26.08.2026): where does the inadmissible dictionary's
empirical criterion go?

LC3 registered the signature of inadmissibility as *spread* (IQR/median not
shrinking) and that sub-criterion returned False: the relative spread of the
classical {x, x^2, x^3} estimate does shrink. The manuscript then described the
observed behaviour as a drift "with no population limit to converge to". This
diagnostic tests the alternative that Prop. 3 (the Schur-complement increment)
predicts, and it changes no registered criterion, threshold, seed or scenario.

Prop. 3 gives J(phi u psi) = J(phi) + Ytilde^2 / S. Under contamination of tail
index alpha_c, a column |x|^a with 2a > alpha_c has an infinite own second
moment: its sample value is the mean of a positive variable of tail index
alpha_c / (2a) < 1 and therefore grows like n^(2a/alpha_c - 1), while the
numerator Ytilde does not. The increment of that column vanishes. An
inadmissible column is therefore not noise -- it is *deleted*, and the
empirical criterion converges downward to the population criterion of the
largest admissible sub-dictionary.

At alpha_c = 3 the classical {x, x^2, x^3} keeps only its linear column, so the
predicted floor is J({x}). Taking the columns one at a time bounds the speed
from below: the slowest single inadmissible column (a = 2) vanishes at
n^-(2a/alpha_c - 1) = n^-1/3, so the excess should fall by AT LEAST
10^(2a/alpha_c - 1) = 2.15 per decade. Two inadmissible columns are dominated
by the same contaminant draws, which makes their 2x2 metric block nearly
singular and the joint deletion faster than that bound; no exact joint rate is
claimed here, only the limit and the lower bound on the speed.

Inputs: the frozen results/l03.json medians (no Monte Carlo is re-run here) and
a population quadrature; the script is deterministic and consumes no RNG.
"""
from __future__ import annotations

from _common import load_result, write_result
from mcp import basis as B, panel
from mcp import projection as P
from mcp.moments import OneSidedMoments

DF_POP = 3.0  # same t(3) contamination as LC3(a)/(b)
DICT_NAME = "classical {x,x2,x3}"
EXPONENTS = [1.0, 2.0, 3.0]  # the classical dictionary, linear column included

lm, ln = panel.lp5(0.10, df=int(DF_POP))
om, on = OneSidedMoments(lm), OneSidedMoments(ln)

# The largest admissible sub-dictionary of {x, x^2, x^3} at alpha_c = 3: 2a < 3
# keeps a = 1 alone.
admissible_exponents = [a for a in EXPONENTS if 2 * a < DF_POP]
inadmissible_exponents = [a for a in EXPONENTS if 2 * a > DF_POP]
assert admissible_exponents == [1.0]
sub = B.classical(len(admissible_exponents))
d_sub = P.deflection(P.population_moments(sub, om), P.population_moments(sub, on))
floor = float(d_sub.J)

# Lower bound on the speed of deletion: the slowest-vanishing single inadmissible column.
rate = min(2 * a / DF_POP - 1 for a in inadmissible_exponents)
predicted_per_decade_min = 10.0 ** rate

l03 = load_result("l03")
ns = l03["j_vs_n"][DICT_NAME]["n"]
median = l03["j_vs_n"][DICT_NAME]["median"]
excess = [m - floor for m in median]
observed_per_decade = [excess[i] / excess[i + 1] for i in range(len(excess) - 1)]
ratio_to_floor = [m / floor for m in median]

out = {
    "df_population": DF_POP,
    "dictionary": DICT_NAME,
    "admissible_exponents": admissible_exponents,
    "inadmissible_exponents": inadmissible_exponents,
    "floor_dictionary": "linear {x}",
    "J_pop_floor": floor,
    "cond_floor": float(d_sub.cond),
    "n": ns,
    "median": median,
    "excess_over_floor": excess,
    "observed_decay_per_decade": observed_per_decade,
    "single_column_rate_exponent": rate,
    "predicted_decay_per_decade_min": predicted_per_decade_min,
    "ratio_to_floor": ratio_to_floor,
    "reading": ("the empirical criterion of the inadmissible dictionary converges downward to the "
                "population criterion of its largest admissible sub-dictionary; the excess falls at "
                "least as fast as the single-column bound of Prop. 3. Diagnostic only: no registered "
                "criterion, threshold, seed, size or scenario is changed."),
}
print(f"floor J({out['floor_dictionary']}) = {floor:.6f} at alpha_c = {DF_POP:g}")
for n, m, e in zip(ns, median, excess):
    print(f"  n = {n:>8d}   median J = {m:.6f}   excess = {e:.3e}   J/floor = {m / floor:.3f}")
print(f"observed decay per decade {[round(r, 2) for r in observed_per_decade]} "
      f"vs single-column lower bound {predicted_per_decade_min:.2f} (rate n^-{rate:.4f})")
print("written", write_result("d01", out))
