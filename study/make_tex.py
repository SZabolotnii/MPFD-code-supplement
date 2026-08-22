#!/usr/bin/env python3
"""Generate paper/generated/{numbers,table1,table2,table3,table4}.tex from study/results.

Only source of manuscript numbers -- every macro here is read straight out of a
frozen results/*.json (see study/checks/_common.py:load_result); nothing is typed
by hand. Re-running build() must reproduce the committed files byte for byte
(checked by verify_reported_values.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "checks"))
from _common import load_result  # noqa: E402

OUT = STUDY.parent / "paper" / "generated"

# l04["pairs"] / l05["oracle"] name -> macro-name key suffix (fixed contract,
# reused verbatim by Task 15).
LP_ONE = "LP1 laplace scale 1 vs 1.3"
LP_TWO = "LP2 symmetric mixtures g4 1.26 vs 2.61"
KEYS = {"LP1 laplace scale 1 vs 1.3": "LPone", "LP2 symmetric mixtures g4 1.26 vs 2.61": "LPtwo",
        "LP3 A2L shift 0.3": "AtwoL", "LP3 A4L shift 0.3": "AfourL", "LP3 A5 shift 0.3": "Afive",
        "LP4 A1L vs A1R": "AoneLR", "LP4 A2L vs reflect(A2L)": "AtwoLmirror"}

# Keys whose free-even dictionary E captures a non-negligible fraction of
# J_inf. AoneLR and AtwoLmirror have E.frac ~ 1e-27 (an even dictionary is
# uninformative on a pair that differs only in the sign of skew), so
# gain_H_over_E_pct there is a ~1e28 ratio-of-noise artifact, not a number
# (R14-3): excluded from \gainHE emission.
EVEN_INFORMATIVE_KEYS = ("LPone", "LPtwo", "AtwoL", "AfourL", "Afive")

# The three LP3 pairs get oracle-vs-path macros (LC5) under the same key
# suffixes used for LC4, so a reader can align Table 1 rows with the oracle
# comparison sentence.
LP3_KEYS = ("AtwoL", "AfourL", "Afive")

FIRST_SPLIT = "train first half / test second"
DRONE_PAIR = "LP5 drone vs pult (eps=0)"

# gain_H_over_M_pct is floating-point noise around an exact tie (H = M) on
# AoneLR (0.0 exactly) and AtwoLmirror (~-8.7e-14); below this magnitude a
# value already rounds to "0.0" at one decimal regardless of sign, so
# snapping it to a clean +0.0 first (I3) only ever changes the glyph shown
# for a genuine tie, never a real number: it turns "-0.0" into "0.0".
GAIN_TIE_EPS = 0.05


def sci_literal(value: float) -> str:
    """One-significant-digit LaTeX scientific literal, e.g. 8.95e-16 -> '9\\times 10^{-16}'."""
    mantissa, exp = f"{value:.0e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exp)}}}"


def _clamp_tie(value: float, eps: float = GAIN_TIE_EPS) -> float:
    """Snap a value within eps of zero to a clean +0.0 (I3) so formatting never
    prints a misleading '-0.0' for a quantity whose content is 'exactly tied'."""
    return 0.0 if abs(value) < eps else value


def _tampere_row_macros(rows: dict, row_key: str, macro_prefix: str) -> list[str]:
    r = rows[row_key]
    return [f"\\newcommand{{\\{macro_prefix}}}{{{r['auc']:.3f}}}",
            f"\\newcommand{{\\{macro_prefix}Lo}}{{{r['ci_lo']:.3f}}}",
            f"\\newcommand{{\\{macro_prefix}Hi}}{{{r['ci_hi']:.3f}}}"]


def _table2_label(row_key: str) -> str:
    """Readable math-mode label for an l06 dictionary row, matched by prefix
    (never the raw string -- the halfline key embeds a Python np.float64 repr)."""
    if row_key.startswith("halfline"):
        return r"half-line $H$"
    if row_key.startswith("linear"):
        return r"linear $\{x\}$"
    if row_key.startswith("classical"):
        return r"classical $\{x,x^2,x^3\}$"
    if row_key.startswith("PATP"):
        astar = row_key.split("=", 1)[1]
        return rf"PATP $a^*={astar}$"
    if row_key.startswith("mixed"):
        astar = row_key.split("=", 1)[1]
        return rf"mixed $a^*={astar}$"
    raise ValueError(f"no display label defined for l06 row key: {row_key!r}")


def _mean_z(l02: dict, pair: str) -> float:
    """Mean displacement of the folded AUC grid above chance, in units of the registered SE."""
    rec = l02["patp_auc"][pair]
    return sum((a - 0.5) / rec["se"] for a in rec["auc"]) / len(rec["auc"])


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    l01 = load_result("l01")
    l02 = load_result("l02")
    l03 = load_result("l03")
    l04 = load_result("l04")
    l05 = load_result("l05")
    l06 = load_result("l06")

    m = []

    # --- LC4: captured fraction J/J_inf at the 3-column budget (Table 1) ---
    for name, key in KEYS.items():
        p = l04["pairs"][name]
        gain_hm = _clamp_tie(p["gain_H_over_M_pct"])
        m.append(f"\\newcommand{{\\gainHM{key}}}{{{gain_hm:.1f}}}")
        for d in ("H", "M", "P"):
            m.append(f"\\newcommand{{\\frac{d}{key}}}{{{p[d]['frac']:.3f}}}")
        m.append(f"\\newcommand{{\\fracE{key}}}{{{p['E']['frac']:.3f}}}")
    for key in EVEN_INFORMATIVE_KEYS:
        name = next(n for n, k in KEYS.items() if k == key)
        p = l04["pairs"][name]
        m.append(f"\\newcommand{{\\gainHE{key}}}{{{p['gain_H_over_E_pct']:+.1f}}}")
        # Unsigned twin for prose that already names the direction ("at most ... %").
        m.append(f"\\newcommand{{\\gainHEabs{key}}}{{{abs(p['gain_H_over_E_pct']):.1f}}}")
    m.append(f"\\newcommand{{\\diagIdentityErr}}{{{sci_literal(l04['diag_identity_max_err'])}}}")

    # --- LC3: contamination collapse / admissibility (l03) ---
    eps = l03["eps"]
    t = l03["auc_table"]
    classical, mixed = t["classical {x,x2,x3}"], t["mixed {x,|x|^.75,sgn|x|^.75}"]
    # eps=0 and eps=0.02 (I5) are LC3's sharpest contrast -- 0.962 vs 1.000
    # unpolluted, then 0.519 vs 0.775 after 2% contamination -- alongside the
    # already-wired eps=0.05/0.2 pair.
    linear, patp_low = t["linear {x}"], t["PATP a=0.31"]
    for label, e in (("Zero", 0.0), ("Two", 0.02), ("Five", 0.05), ("Twenty", 0.2)):
        i = eps.index(e)
        m.append(f"\\newcommand{{\\aucClassicalAtEps{label}}}{{{classical[i]['auc']:.3f}}}")
        m.append(f"\\newcommand{{\\aucMixedAtEps{label}}}{{{mixed[i]['auc']:.3f}}}")
        # The linear rule and the sub-unit PATP are the honest comparators for the
        # contamination figure (review R2): every dictionary whose exponents stay at
        # or below one survives, so the mixed dictionary's margin over them is the
        # number a reader needs next to its margin over the cubic rule.
        m.append(f"\\newcommand{{\\aucLinearAtEps{label}}}{{{linear[i]['auc']:.3f}}}")
        m.append(f"\\newcommand{{\\aucPatpLowAtEps{label}}}{{{patp_low[i]['auc']:.3f}}}")
    # Hill-grid shape (review R3): J of M({a}) decreases from the lower grid edge up to
    # the last admissible sub-unit point, so the argmax at the edge is censored by the grid.
    hill = l03["hill"]
    hg, hJ = hill["grid"], hill["J"]
    sub_unit = [(a, j) for a, j in zip(hg, hJ) if a < 1.0 and j == j]
    assert all(sub_unit[i + 1][1] < sub_unit[i][1] for i in range(len(sub_unit) - 1)), \
        "hill J is no longer monotone below a=1; fix the manuscript sentence"
    m += [f"\\newcommand{{\\hillGridLo}}{{{hg[0]:.1f}}}",
          f"\\newcommand{{\\hillGridHi}}{{{hg[-1]:.1f}}}",
          f"\\newcommand{{\\hillMonotoneUpTo}}{{{sub_unit[-1][0]:.1f}}}"]
    m += [f"\\newcommand{{\\hillAlpha}}{{{l03['hill']['alpha_hat']:.2f}}}",
          f"\\newcommand{{\\hillCeiling}}{{{l03['hill']['ceiling']:.2f}}}",
          f"\\newcommand{{\\argmaxA}}{{{l03['hill']['argmax_a']:.2f}}}",
          f"\\newcommand{{\\equivMaxDAUC}}{{{l03['equivalence']['max_abs_dAUC']:.3f}}}",
          f"\\newcommand{{\\classicalMedianCollapse}}{{{l03['diagnostics']['classical_median_collapse']:.2f}}}",
          f"\\newcommand{{\\relErrMixedLargest}}{{{l03['diagnostics']['rel_err_mixed'][-1]:.3f}}}",
          f"\\newcommand{{\\equivWithinCells}}{{{l03['diagnostics']['equivalence_within_ci']['n_within']}}}",
          f"\\newcommand{{\\equivTotalCells}}{{{l03['diagnostics']['equivalence_within_ci']['n_cells']}}}"]

    # --- LC5: oracle vs feasible exponent selection (l05) ---
    o, f = l05["oracle"][DRONE_PAIR], l05["feasible"][DRONE_PAIR]
    # feasMeanP/Free/Mixed at 4 decimals (I4): all three round to 1.000 at 3
    # decimals even though they are genuinely different (l05's own paired-CI
    # fields resolve the gaps), which reads as a copy-paste bug.
    m += [f"\\newcommand{{\\oracleGainFreeSignedDrone}}{{{o['gain_free_signed_pct']:+.1f}}}",
          f"\\newcommand{{\\oracleGainMixedDrone}}{{{o['gain_mixed_pct']:+.1f}}}",
          f"\\newcommand{{\\feasMeanP}}{{{f['P']['auc_mean']:.4f}}}",
          f"\\newcommand{{\\feasMeanFree}}{{{f['free_signed']['auc_mean']:.4f}}}",
          f"\\newcommand{{\\feasMeanMixed}}{{{f['mixed']['auc_mean']:.4f}}}"]
    # Paired differences (fix round 2 / item 2): the three means above agree
    # at 4 decimals not because of missing precision but because the
    # feasible arm is saturated at N=256 (CHECKS.md's own post-run
    # annotation) -- the actual citable finding lives in these paired CIs,
    # which cancel the common across-seed variation the raw means share.
    # Units are 1e-5 throughout (values would be all-but-illegible at 3-4
    # decimals otherwise). The manuscript may NOT write the unit as braced
    # scientific notation: the gate's SCI_PATTERN matches 10^{-5} and a
    # '% fixed:' marker cannot exempt it (the marker grammar parses decimal
    # literals only), so that form is a guaranteed gate failure. Write the
    # unit without braces instead -- main.tex says "in parts in $10^5$ of AUC".
    diff_free = f["diff_free_signed_minus_P"]
    diff_mixed = f["diff_mixed_minus_P"]
    m.append(r"% feasDiff{Free,Mixed}MinusP{,Lo,Hi} below are in units of 1e-5;"
              r" write the unit without braced scientific notation (the gate rejects it),"
              r" e.g. \feasDiffFreeMinusP in parts in $10^5$ of AUC")
    m += [f"\\newcommand{{\\feasDiffFreeMinusP}}{{{diff_free['mean'] * 1e5:+.2f}}}",
          f"\\newcommand{{\\feasDiffFreeMinusPLo}}{{{diff_free['ci_lo'] * 1e5:+.2f}}}",
          f"\\newcommand{{\\feasDiffFreeMinusPHi}}{{{diff_free['ci_hi'] * 1e5:+.2f}}}",
          f"\\newcommand{{\\feasDiffMixedMinusP}}{{{diff_mixed['mean'] * 1e5:+.2f}}}",
          f"\\newcommand{{\\feasDiffMixedMinusPLo}}{{{diff_mixed['ci_lo'] * 1e5:+.2f}}}",
          f"\\newcommand{{\\feasDiffMixedMinusPHi}}{{{diff_mixed['ci_hi'] * 1e5:+.2f}}}"]
    for key in LP3_KEYS:
        name = next(n for n, k in KEYS.items() if k == key)
        oo = l05["oracle"][name]
        m.append(f"\\newcommand{{\\oracleGainFreeSigned{key}}}{{{oo['gain_free_signed_pct']:+.1f}}}")
        m.append(f"\\newcommand{{\\oracleGainMixed{key}}}{{{oo['gain_mixed_pct']:+.1f}}}")

    # --- LC2: parity split and PATP-AUC-at-chance (l02) ---
    parity_lp2 = l02["parity"]["LP2 symmetric mixtures g4 1.26 vs 2.61"]
    patp_lp1 = l02["patp_auc"]["LP1 laplace scale 1 vs 1.3"]
    patp_lp2 = l02["patp_auc"]["LP2 symmetric mixtures g4 1.26 vs 2.61"]
    m += [f"\\newcommand{{\\JoddLPtwo}}{{{sci_literal(parity_lp2['J_odd'])}}}",
          f"\\newcommand{{\\maxPatpAucLPone}}{{{patp_lp1['max_auc']:.5f}}}",
          f"\\newcommand{{\\maxPatpAucLPtwo}}{{{patp_lp2['max_auc']:.5f}}}",
          f"\\newcommand{{\\patpAucSE}}{{{patp_lp1['se']:.5f}}}",
          f"\\newcommand{{\\nAlphaGrid}}{{{len(l02['alphas'])}}}",
          # Folded-AUC displacement of the whole alpha grid (LC2's registered failure): the
          # stored AUCs are the folded max(a,1-a), so report their mean offset in SE units.
          f"\\newcommand{{\\foldedMeanZLPone}}{{{_mean_z(l02, LP_ONE):.2f}}}",
          f"\\newcommand{{\\foldedMeanZLPtwo}}{{{_mean_z(l02, LP_TWO):.2f}}}"]

    # --- LC1: algebraic identities (l01) ---
    m += [f"\\newcommand{{\\identityErr}}{{{sci_literal(l01['identity_max_abs_err'])}}}",
          f"\\newcommand{{\\schurErr}}{{{sci_literal(l01['schur_check'])}}}",
          f"\\newcommand{{\\mixedHalflineErr}}{{{sci_literal(l01['mixed_vs_halfline_err'])}}}"]

    # --- LC6: Tampere leakage-safe split (l06), first split ---
    split0 = l06["splits"][FIRST_SPLIT]
    rows0 = split0["rows"]
    m += _tampere_row_macros(rows0, "linear {x}", "tampereAucLinear")
    m += _tampere_row_macros(rows0, "classical {x,x2,x3}", "tampereAucClassical")
    mixed_key = next(k for k in rows0 if k.startswith("mixed"))
    m += _tampere_row_macros(rows0, mixed_key, "tampereAucMixed")
    m += [f"\\newcommand{{\\tampereAlphaHat}}{{{split0['hill_alpha_hat']:.2f}}}",
          f"\\newcommand{{\\tampereNRecordings}}{{{l06['n_recordings_per_class']}}}"]

    (out_dir / "numbers.tex").write_text("% generated by study/make_tex.py -- do not edit\n" + "\n".join(m) + "\n")

    # --- Table 1: captured fraction J/J_inf at the 3-column budget (LC4) ---
    # Gains are unsigned in numbers.tex (prose reads "H gains 244.8% over M")
    # but signed here (tables conventionally show a signed delta column) --
    # deliberate (Mn1), not a formatting inconsistency.
    # E (the free even pair) is a column here, not only a macro: the manuscript's
    # central attribution claim -- that H's margin over M is a parity-budget
    # artefact rather than the support split -- is read off H against E, so a
    # reader must be able to check it in the table (R1-1).
    rows = ["% generated by study/make_tex.py -- do not edit",
            "\\begin{tabular}{lcccccc}", "\\toprule",
            "pair & $J_\\infty$ & $H$ & $M$ & $P$ & $E$ & gain $H/M$, \\% \\\\", "\\midrule"]
    for name, key in KEYS.items():
        p = l04["pairs"][name]
        gain_hm = _clamp_tie(p["gain_H_over_M_pct"])
        rows.append(f"{name} & {p['J_inf']:.3f} & {p['H']['frac']:.3f} & {p['M']['frac']:.3f} "
                    f"& {p['P']['frac']:.3f} & {p['E']['frac']:.3f} & {gain_hm:+.1f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table1.tex").write_text("\n".join(rows) + "\n")

    # --- Table 2: Tampere first-split rows, all five dictionaries (LC6) ---
    t2 = ["% generated by study/make_tex.py -- do not edit",
          "\\begin{tabular}{lccc}", "\\toprule", "dictionary & AUC & 95\\% CI & cond \\\\", "\\midrule"]
    for row_key, r in rows0.items():
        label = _table2_label(row_key)
        t2.append(f"{label} & {r['auc']:.3f} & [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] & {r['cond']:.1f} \\\\")
    t2 += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table2.tex").write_text("\n".join(t2) + "\n")

    # --- Table 3: where the LC4 optima sit (review R3) ---
    # The exponents a*, b* at which each dictionary attains its Table-1 fraction, with
    # cond(F) there; an optimum on an edge of the pre-registered grid is marked with a
    # dagger, because the fraction (and hence every gain in the prose) is then censored
    # by the grid rather than attained. P is searched on its own alpha grid.
    grid = l04["grid"]
    alphas = l04["alphas"]
    lo, hi = min(grid), max(grid)
    alo, ahi = min(alphas), max(alphas)

    def _fmt_args(args, edge_lo, edge_hi):
        parts = []
        for a in args:
            mark = r"$^{\dagger}$" if abs(a - edge_lo) < 1e-9 or abs(a - edge_hi) < 1e-9 else ""
            parts.append(f"{a:.2f}{mark}")
        return ", ".join(parts)

    n_edge, n_cells = 0, 0
    t3 = ["% generated by study/make_tex.py -- do not edit",
          "\\begin{tabular}{lcccccccc}", "\\toprule",
          "pair & \\multicolumn{2}{c}{$H$: $a^*,b^*$ / cond} & \\multicolumn{2}{c}{$M$: $a^*$ / cond}"
          " & \\multicolumn{2}{c}{$P$: $\\alpha^*$ / cond} & \\multicolumn{2}{c}{$E$: $a^*,b^*$ / cond} \\\\",
          "\\midrule"]
    for name, key in KEYS.items():
        p = l04["pairs"][name]
        cells = []
        for d in ("H", "M", "E"):
            args = p[d]["args"]
            n_cells += 1
            n_edge += int(any(abs(a - lo) < 1e-9 or abs(a - hi) < 1e-9 for a in args))
            cells.append((d, _fmt_args(args, lo, hi), p[d]["cond"]))
        pa = p["P"]["args"]
        n_cells += 1
        n_edge += int(any(abs(a - alo) < 1e-9 or abs(a - ahi) < 1e-9 for a in pa))
        cells.insert(2, ("P", _fmt_args(pa, alo, ahi), p["P"]["cond"]))
        row = name
        for _, args_s, cond in cells:
            row += f" & {args_s} & {cond:.0f}"
        t3.append(row + " \\\\")
    t3 += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table3.tex").write_text("\n".join(t3) + "\n")
    # Appended to numbers.tex after the fact so the count is emitted by the same
    # generator that prints the table it summarises.
    with (out_dir / "numbers.tex").open("a") as fh:
        fh.write(f"\\newcommand{{\\nEdgeCells}}{{{n_edge}}}\n\\newcommand{{\\nOptCells}}{{{n_cells}}}\n")

    # --- Table 4: the feasible arm of LC5 in full (moved out of the main text) ---
    # Mean test AUC per dictionary and the paired differences against P with their
    # 95% intervals over seeds, in parts in 1e5 of AUC (same unit as the macros).
    t4 = ["% generated by study/make_tex.py -- do not edit",
          "\\begin{tabular}{lccccc}", "\\toprule",
          "pair & $P$ & free odd & mixed & free odd $-$ $P$ & mixed $-$ $P$ \\\\", "\\midrule"]
    for name in list(l05["feasible"]):
        fz = l05["feasible"][name]
        df_, dm_ = fz["diff_free_signed_minus_P"], fz["diff_mixed_minus_P"]
        t4.append(f"{name} & {fz['P']['auc_mean']:.4f} & {fz['free_signed']['auc_mean']:.4f} & {fz['mixed']['auc_mean']:.4f}"
                  f" & {df_['mean'] * 1e5:+.2f} [{df_['ci_lo'] * 1e5:+.2f}, {df_['ci_hi'] * 1e5:+.2f}]"
                  f" & {dm_['mean'] * 1e5:+.2f} [{dm_['ci_lo'] * 1e5:+.2f}, {dm_['ci_hi'] * 1e5:+.2f}] \\\\")
    t4 += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table4.tex").write_text("\n".join(t4) + "\n")


if __name__ == "__main__":
    build(OUT)
    print("written", OUT / "numbers.tex", OUT / "table1.tex", OUT / "table2.tex", OUT / "table3.tex", OUT / "table4.tex")
