"""Fixture-pinned macros and gate-behavior probes for make_tex.py / verify_reported_values.py.

The gate's own staleness check (verify_reported_values.run_checks) regenerates via
make_tex.build() and diffs the result against make_tex.build()'s own committed output --
by construction it can never catch a wrong-but-deterministic JSON->macro mapping inside
build() itself (I6). The PINNED_MACROS test below is the actual defense against that: each
literal was computed by hand from study/results/l0N.json (see the comment above each one),
independently of make_tex.py, then pinned here -- NOT re-derived via make_tex.py's own
helpers (sci_literal, _clamp_tie, ...), since a test that reuses the code under test proves
only that the code agrees with itself.

The GATE PROBE tests below exercise verify_reported_values.run_checks() against a tmp_path
sandbox laid out like paper/, one test per bypass class the fix rounds closed
(I1, I2, Mn2's generated-input-macro half, Mn3), several covering the '% fixed: <values>'
value-enumeration mechanism (fix round 2 / item 1: exempts exactly the listed values, not the
whole line -- see test_gate_fixed_marker_* below), plus regression tests for the legitimate
exemptions that must keep working and two staleness probes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import make_tex
from verify_reported_values import run_checks

# Each literal was independently confirmed against study/results/l0N.json
# before being pinned (values as of git HEAD; see task-14 fix-round report
# for the by-hand derivation of every one of these):
#   LC1 (l01):    identity_max_abs_err = 8.951173136040325e-16
#                 -> 1 significant digit -> 9e-16
#   LC2 (l02):    len(alphas) = 37
#                 LP1 patp_auc.max_auc = 0.5110385 -> .5f rounds UP to 0.51104
#   LC3 (l03):    classical{x,x2,x3} auc at eps=0.0  = 0.96199625  -> .3f 0.962 (I5, new)
#                 mixed{...} auc at eps=0.02          = 0.7752943125 -> .3f 0.775 (I5, new)
#   LC4 (l04):    LP4 A2L-vs-mirror gain_H_over_M_pct = -8.721112976537642e-14;
#                 |v| < 0.05 -> clamped to a clean 0.0, not "-0.0" (I3)
#                 LP2 E.frac = 0.774704762755471 -> .3f 0.775
#   LC5 (l05):    drone-pair feasible mixed auc_mean  = 0.9997307416666666
#                 -> .4f 0.9997 (I4; 3 decimals would have printed "1.000")
#                 LP3-A4L oracle gain_mixed_pct        = 43.82347760532585 -> +.1f +43.8
#                 diff_free_signed_minus_P.mean  = -3.8258333333333434e-05
#                 -> *1e5 = -3.8258... -> +.2f -3.83 (fix round 2 / item 2)
#                 diff_mixed_minus_P.mean        = -6.780833333334125e-05
#                 -> *1e5 = -6.7808... -> +.2f -6.78 (fix round 2 / item 2)
#   LC6 (l06):    first-split hill_alpha_hat = 7.289726508292674 -> .2f 7.29
#                 n_recordings_per_class = 1 (int)
PINNED_MACROS = [
    r"\newcommand{\identityErr}{9\times 10^{-16}}",             # LC1
    r"\newcommand{\nAlphaGrid}{37}",                             # LC2
    r"\newcommand{\maxPatpAucLPone}{0.51104}",                   # LC2
    r"\newcommand{\aucClassicalAtEpsZero}{0.962}",               # LC3 (I5)
    r"\newcommand{\aucMixedAtEpsTwo}{0.775}",                    # LC3 (I5)
    r"\newcommand{\aucLinearAtEpsTwo}{0.611}",                   # LC3 (review R2)
    r"\newcommand{\hillMonotoneUpTo}{0.9}",                      # LC3 (review R3)
    r"\newcommand{\nEdgeCells}{10}",                             # LC4 (review R3)
    r"\newcommand{\gainHMAtwoLmirror}{0.0}",                     # LC4 (I3)
    r"\newcommand{\fracELPtwo}{0.775}",                          # LC4
    r"\newcommand{\feasMeanMixed}{0.9997}",                      # LC5 (I4)
    r"\newcommand{\oracleGainMixedAfourL}{+43.8}",               # LC5
    r"\newcommand{\feasDiffFreeMinusP}{-3.83}",                  # LC5 (fix round 2 / item 2)
    r"\newcommand{\feasDiffMixedMinusP}{-6.78}",                 # LC5 (fix round 2 / item 2)
    r"\newcommand{\tampereAlphaHat}{7.29}",                      # LC6
    r"\newcommand{\tampereNRecordings}{1}",                      # LC6
]


@pytest.fixture(scope="module")
def numbers_tex_lines(tmp_path_factory) -> list[str]:
    out = tmp_path_factory.mktemp("generated")
    make_tex.build(out)
    return (out / "numbers.tex").read_text().splitlines()


@pytest.mark.parametrize("expected_line", PINNED_MACROS)
def test_pinned_macro_line(numbers_tex_lines, expected_line):
    assert expected_line in numbers_tex_lines


# ---------------------------------------------------------------------------
# Gate probes: verify_reported_values.run_checks() against a tmp_path sandbox
# ---------------------------------------------------------------------------

@pytest.fixture
def gate_root(tmp_path) -> Path:
    """A tmp_path laid out like paper/: a freshly-built (non-stale)
    generated/ dir, ready for a test to drop main.tex content into."""
    make_tex.build(tmp_path / "generated")
    return tmp_path


def test_gate_passes_on_empty_manuscript(gate_root):
    (gate_root / "main.tex").write_text("")
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_catches_one_decimal_hardcode(gate_root, capsys):
    """I1: the regex used to require >=2 fractional digits, so one-decimal
    headline percentages like this one slipped through."""
    (gate_root / "main.tex").write_text("The H dictionary gains 244.8% over M on LP2.\n")
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "HARD-CODED NUMBER main.tex:1" in capsys.readouterr().out


def test_gate_catches_bare_scientific_notation(gate_root, capsys):
    """I2: a hand-typed sci-notation literal has no digit.digit substring and
    used to be invisible to the linter."""
    (gate_root / "main.tex").write_text("The identity error is 9\\times 10^{-16} in magnitude.\n")
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "HARD-CODED NUMBER main.tex:1" in capsys.readouterr().out


def test_gate_catches_bare_exponent_without_times(gate_root, capsys):
    """I2's second half: a bare 10^{...} (no preceding \\times) must also be caught."""
    (gate_root / "main.tex").write_text("A magnitude of 10^{-16} was observed.\n")
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "HARD-CODED NUMBER main.tex:1" in capsys.readouterr().out


def test_gate_catches_number_co_located_with_generated_input(gate_root, capsys):
    """Mn2: '"generated/" in line' used to exempt the whole line, so an
    unrelated number riding along with a legitimate \\input escaped."""
    (gate_root / "main.tex").write_text("\\input{generated/numbers} and separately AUC = 12.345 here.\n")
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "12.345" in capsys.readouterr().out


def test_gate_catches_number_co_located_with_valueless_fixed_marker(gate_root, capsys):
    """Mn2, round 1: '"% fixed:" in line' used to exempt the whole line by
    substring match alone. A marker with no numeric value after 'fixed:' (as
    this one has none) exempts nothing under the round-2 mechanism either --
    it fails to match FIXED_MARKER at all, so the whole line is linted."""
    (gate_root / "main.tex").write_text(
        "Totally unrelated result 55.123 on this line. % fixed: exponent-grid endpoint\n"
    )
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "55.123" in capsys.readouterr().out


def test_gate_fixed_marker_exempts_the_value_it_lists(gate_root):
    """Fix round 2 / item 1: round 1's fix (strip the trailing comment, lint
    the rest) could never work, because everything after an unescaped '%' is
    a LaTeX comment -- a marker can only ever follow the constant it exempts,
    so there was no text left to lint once the comment was stripped. The
    corrected mechanism: '% fixed: 1.95' ENUMERATES the value(s) it exempts,
    matched as exact strings against what NUMBER_PATTERN finds before the
    marker."""
    (gate_root / "main.tex").write_text("The grid endpoint is 1.95. % fixed: 1.95\n")
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_fixed_marker_does_not_exempt_an_unrelated_number(gate_root, capsys):
    """The enumerated value list is not a whole-line exemption: a listed
    value ('0.6') does not excuse a different, unrelated number ('55.123')
    also present on the line."""
    (gate_root / "main.tex").write_text(
        "Totally unrelated result 55.123 on this line. % fixed: 0.6\n"
    )
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "55.123" in capsys.readouterr().out


def test_gate_fixed_marker_allows_multiple_comma_separated_values(gate_root):
    (gate_root / "main.tex").write_text(
        "We use the grid $[0.55, 1.95]$ (15 points) % fixed: 0.55, 1.95\n"
    )
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_fixed_marker_allows_free_prose_after_the_value_list(gate_root):
    (gate_root / "main.tex").write_text(
        "\\includegraphics[width=0.6\\linewidth]{figs/fig1_diagnostics} % fixed: 0.6 -- figure width\n"
    )
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_catches_nested_file(gate_root, capsys):
    """Mn3: a non-recursive glob missed anything below paper/*.tex."""
    (gate_root / "main.tex").write_text("")
    supp = gate_root / "supp"
    supp.mkdir()
    (supp / "main.tex").write_text("Nested supplementary claim: 44.888 improvement.\n")
    assert run_checks(gate_root, gate_root / "generated") is False
    assert "44.888" in capsys.readouterr().out


def test_gate_never_lints_the_generated_dir_itself(gate_root):
    """generated/*.tex is full of legitimate decimals (that is its job); the
    recursive walk added for Mn3 must still exclude it."""
    (gate_root / "main.tex").write_text("")
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_allows_legitimate_generated_input_line(gate_root):
    (gate_root / "main.tex").write_text(
        "\\input{generated/numbers} % contains many decimals, not linted itself\n"
    )
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_allows_full_comment_line(gate_root):
    (gate_root / "main.tex").write_text("% a comment with 12.345 should be skipped\n")
    assert run_checks(gate_root, gate_root / "generated") is True


def test_gate_detects_stale_generated_file(gate_root):
    numbers = gate_root / "generated" / "numbers.tex"
    original = numbers.read_text()
    perturbed = original.replace(r"\newcommand{\hillAlpha}{", r"\newcommand{\hillAlpha}{9", 1)
    assert perturbed != original  # sanity: the macro exists and was actually perturbed
    numbers.write_text(perturbed)
    (gate_root / "main.tex").write_text("")
    assert run_checks(gate_root, gate_root / "generated") is False


def test_gate_detects_missing_generated_file(gate_root):
    (gate_root / "generated" / "table2.tex").unlink()
    (gate_root / "main.tex").write_text("")
    assert run_checks(gate_root, gate_root / "generated") is False


def test_gate_ignores_hidden_tool_directories(gate_root):
    """A flattened copy under a hidden directory (e.g. /papermentor's .cache) is tool
    state, not manuscript prose: linting it re-reports every generated decimal."""
    cache = gate_root / ".papermentor" / ".cache"
    cache.mkdir(parents=True)
    (cache / "merged.tex").write_text("row & 0.216 & 0.913 \\\\\n")
    assert run_checks(gate_root, gate_root / "generated") is True
