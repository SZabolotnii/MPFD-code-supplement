#!/usr/bin/env python3
"""Figures for the Letter from study/results/*.json only.

Three separate figures, not two with panels: Elsevier's artwork instructions
forbid combining different graphs into one image, so the diagnostic curves and
the captured-fraction bars are emitted as Fig. 1 and Fig. 2 and the
contamination curves as Fig. 3, each its own file.

Fig. 1 shows three representative diagnostic pairs, not all eight in
l02["curves"] (that gave 16 overlapping lines with duplicate legend entries in
a 3.5-inch panel): LP2 (symmetric mixtures, informative only in the
even/dispersion diagnostic), LP3 A2L (both diagnostics informative), and LP4
A1L-vs-A1R (mirror-skew pair, informative only in the odd diagnostic).
Fig. 2 shows four dictionaries per pair -- H, M, P, E -- because the
manuscript's central finding is that the free even pair E matches or beats H on
the symmetric pairs (LP1, LP2); a 3-bar panel would hide that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY / "checks"))
from _common import load_result  # noqa: E402

FIGS = STUDY.parent / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "figure.dpi": 200})

# ---------------------------------------------------------------------------
# Fig. 1: diagnostic curves; Fig. 2: captured fraction of H/M/P/E
# ---------------------------------------------------------------------------
l02, l04 = load_result("l02"), load_result("l04")
# Authored at the manuscript's final printed width (I9/F3): \linewidth in
# elsarticle[preprint,12pt] is 390.0 pt = 5.42 in, so a 7.0 in canvas was being
# scaled to 0.77 by \includegraphics[width=\linewidth] and its 8 pt type printed
# at 6.2 pt against 12 pt body text. Authoring at the printed width keeps the
# type at its nominal size; the aspect ratio is relaxed a little (2.0 instead of
# 2.33) so the outside-axes legend and the rotated tick labels keep their room.
# Width is set so the tight-trimmed PDF matches \linewidth (390 pt) unscaled;
# each figure keeps that width and takes the height one panel needs.
fig1, a1 = plt.subplots(figsize=(5.45, 2.7))

# Fig. 1 (R14-1): Delta-sigma=0 (LP2), both-informative (LP3 A2L), and
# Delta-nu=0 (LP4 A1L-vs-A1R) cases -- the three the paper argues from. Same
# color per pair, solid = Delta-nu, dashed = Delta-sigma, so the reader can
# tell at a glance which diagnostic carries the signal for each pair.
#
# LP2's Delta-sigma and LP4-mirror's Delta-nu are not just small, they are
# identically zero (~1e-16, machine noise) across the whole q-grid -- and
# that vanishing IS the panel's argument (parity means one diagnostic can
# carry all the information while the other carries exactly none). Plotted
# as a plain line, a curve pinned to 0 is invisible: indistinguishable from
# the grey zero-reference line and from each other. Give those two curves
# sparse, distinctly-shaped markers on top of the (still perfectly flat,
# never offset) line, so the reader can see each one was measured and is
# exactly zero, not omitted.
# Both zero curves share the same q-grid, so a common marker stride would put
# their markers on identical coordinates and the later-drawn one would hide the
# other. Stagger the phase as well as the shape: (offset, stride).
ZERO_CURVE_MARKER = {
    ("LP2 symmetric mixtures g4 1.26 vs 2.61", "d_sigma"): ("o", (0, 4)),
    ("LP4 A1L vs A1R", "d_nu"): ("s", (2, 4)),
}
LEFT_PAIRS = (
    ("LP2 symmetric mixtures g4 1.26 vs 2.61", "LP2 sym."),
    ("LP3 A2L shift 0.3", "LP3 A2L"),
    ("LP4 A1L vs A1R", "LP4 mirror"),
)
for i, (name, label) in enumerate(LEFT_PAIRS):
    c = l02["curves"][name]
    q = np.array(c["q"])
    color = f"C{i}"
    for field, ls, sym in (("d_nu", "-", "Δν"), ("d_sigma", "--", "Δσ")):
        spec = ZERO_CURVE_MARKER.get((name, field))
        if spec:
            marker, markevery = spec
            a1.plot(q, c[field], lw=1.0, color=color, ls=ls, marker=marker,
                     markevery=markevery, markersize=4.5, markerfacecolor="none",
                     markeredgewidth=1.1, label=f"{label} {sym}")
        else:
            a1.plot(q, c[field], lw=1.2, color=color, ls=ls, label=f"{label} {sym}")
a1.axhline(0, color="grey", lw=0.6)
a1.set_xlabel("q")
a1.set_ylabel("Δν(q), Δσ(q)")
a1.legend(ncol=2, fontsize=6)
fig1.tight_layout()
fig1.savefig(FIGS / "fig1_diagnostics.pdf", bbox_inches="tight")
fig1.savefig(FIGS / "fig1_diagnostics.png", bbox_inches="tight")
plt.close(fig1)

fig2, a2 = plt.subplots(figsize=(5.45, 2.7))

# Fig. 2 (R14-2): four bars per pair -- H, M, P, E -- at the 3-column
# budget. P.frac is ~1e-27 on LP1/LP2 (an admissible odd dictionary applied
# to a symmetric-vs-symmetric pair correctly captures ~nothing) and E.frac
# is ~1e-27 on both LP4 pairs (an even dictionary is uninformative on a pair
# that differs only in the sign of skew): both draw as no bar at all, which
# reads as "not computed" rather than "computed, and it is zero". Fixed with
# a short baseline tick in the series' own color at each such cell -- an
# annotation only: the true (invisible) value is still what is plotted, not
# rescaled or clipped.
# Bars span nearly the full [0, 1.02] range across most of the seven groups,
# so there is no empty pocket inside the axes for a legend -- it goes in a
# single row above the axes instead. Short single-line tick labels (rotated)
# replace two-line ones, which collided with their neighbors once four bars
# per group left less horizontal room per pair than the original three did.
# Tick labels shortened with the canvas (I9/F3): at 5.08 in the two-token names
# ("LP4 A2L/mirror") collided with their neighbours; the panel prefix is in the
# axis caption, so the pair id alone identifies the group.
SHORT_NAME = {
    "LP1 laplace scale 1 vs 1.3": "LP1",
    "LP2 symmetric mixtures g4 1.26 vs 2.61": "LP2",
    "LP3 A2L shift 0.3": "LP3 A2L",
    "LP3 A4L shift 0.3": "LP3 A4L",
    "LP3 A5 shift 0.3": "LP3 A5",
    "LP4 A1L vs A1R": "LP4 A1L/A1R",
    "LP4 A2L vs reflect(A2L)": "LP4 mirror",
}
# A bar at or below this is drawn as a baseline tick rather than omitted, so a measured
# zero is visible as a zero. The same pairs are the ones make_tex.py leaves out of
# \gainHE (there by an explicit key list, EVEN_INFORMATIVE_KEYS, not by this cutoff).
ZERO_BAR_THRESHOLD = 1e-6
ZERO_TICK_Y = 0.015
names = list(l04["pairs"])
x = np.arange(len(names))
w = 0.2
BARS = (("H", -1.5 * w, "half-line H"), ("M", -0.5 * w, "mixed parity M"),
        ("P", 0.5 * w, "PATP P (odd)"), ("E", 1.5 * w, "free even E"))
for k, off, lab in BARS:
    vals = [l04["pairs"][n][k]["frac"] for n in names]
    bar_container = a2.bar(x + off, vals, w, label=lab)
    tick_color = bar_container.patches[0].get_facecolor()
    for xi, v in zip(x, vals):
        if v < ZERO_BAR_THRESHOLD:
            a2.plot([xi + off - 0.4 * w, xi + off + 0.4 * w], [ZERO_TICK_Y, ZERO_TICK_Y],
                     color=tick_color, lw=2.2, solid_capstyle="butt", zorder=5)
a2.set_xticks(x)
a2.set_xticklabels([SHORT_NAME[n] for n in names], rotation=0, ha="center", fontsize=7)
a2.set_ylabel("J / J∞ (3 columns)")
a2.set_ylim(0, 1.02)
a2.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig2.tight_layout()
fig2.savefig(FIGS / "fig2_budget.pdf", bbox_inches="tight")
fig2.savefig(FIGS / "fig2_budget.png", bbox_inches="tight")
plt.close(fig2)

# ---------------------------------------------------------------------------
# Fig. 3: contamination collapse of integer dictionaries (LC3)
# ---------------------------------------------------------------------------
# Legend labels are mathtext, not raw dict keys (study/results/l03.json's
# auc_table keys, e.g. "mixed {x,|x|^.75,sgn|x|^.75}", are Python
# dictionary-builder syntax, not meant to be read verbatim).
LEGEND_LABEL = {
    "linear {x}": r"linear $\{x\}$",
    "classical {x,x2}": r"classical $\{x, x^2\}$",
    "classical {x,x2,x3}": r"classical $\{x, x^2, x^3\}$",
    "PATP a=0.99": r"PATP $a=0.99$",
    "PATP a=0.31": r"PATP $a=0.31$",
    "mixed {x,|x|^.75,sgn|x|^.75}": r"mixed $\{x, |x|^{0.75}, \mathrm{sgn}(x)|x|^{0.75}\}$",
}
l03 = load_result("l03")
fig3, ax = plt.subplots(figsize=(3.4, 2.6))
for k, rows in l03["auc_table"].items():
    a = [r["auc"] for r in rows]
    lo = [r["ci_lo"] for r in rows]
    hi = [r["ci_hi"] for r in rows]
    ax.plot(l03["eps"], a, marker="o", ms=3, lw=1.2, label=LEGEND_LABEL[k])
    ax.fill_between(l03["eps"], lo, hi, alpha=0.15)
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.set_xlabel("contamination fraction ε")
ax.set_ylabel("AUC (N = 256)")
ax.set_ylim(0.45, 1.02)
ax.legend()
fig3.tight_layout()
fig3.savefig(FIGS / "fig3_contamination.pdf")
fig3.savefig(FIGS / "fig3_contamination.png")
plt.close(fig3)

print("written", sorted(p.name for p in FIGS.iterdir()))
