#!/usr/bin/env python3
"""Gate: manuscript numbers must come from generated/*.tex, which must match study/results.

Two checks, both run by run_checks() so they can be exercised from tests
(study/tests/test_make_tex.py) without going through the CLI:
1. Regenerate numbers.tex/table1.tex/table2.tex/table3.tex/table4.tex into a temp dir and diff them
   against the committed generated/ files -- STALE if they differ or are missing.
2. Recursively lint every .tex file under the manuscript root, excluding the
   generated/ fragment library, for hard-coded numbers or bare scientific
   notation that should be macros instead.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import make_tex  # noqa: E402
from make_tex import OUT, build  # noqa: E402

GENERATED_FILES = ("numbers.tex", "table1.tex", "table2.tex", "table3.tex", "table4.tex")

#: (manuscript root, generated dir, generator, files) -- paper/ and paper/paper_ua/ are
#: both under the root, so the Ukrainian mirror is linted by the same walk.
TARGETS = ((make_tex.OUT.parent, make_tex.OUT, make_tex.build, GENERATED_FILES),)

# A run of digits, a literal dot, then >=1 more digit, not glued to a
# preceding word character or backslash -- so it fires on prose like "AUC =
# 0.95" but not on a LaTeX token such as \S2.1 or an identifier like v2.10.
# Widened from >=2 fractional digits (I1): one-decimal headline percentages
# like "244.8" or "-16.4" must be caught too, so a plain fixed constant like
# 0.6\linewidth now needs an explicit '% fixed:' marker -- intended.
NUMBER_PATTERN = re.compile(r"(?<![\w\\])\d+\.\d+")

# Bare or \times-prefixed scientific-notation literal, e.g. "10^{-16}" or
# "9\times 10^{-16}" (I2): the lone mantissa digit would not match
# NUMBER_PATTERN, so a hand-typed \diagIdentityErr bypass needs its own check.
SCI_PATTERN = re.compile(r"10\^\{[+-]?\d+\}")

# A '% fixed: <values>' marker ENUMERATES the literal decimal values it
# exempts on that line, comma-separated, with optional free prose after the
# last one (fix round 2 / item 1: "strip the trailing comment" alone cannot
# work, because everything after an unescaped '%' is a LaTeX comment, so a
# marker can only ever follow the constant it is meant to exempt -- there is
# no text left to "lint" once the comment is stripped). '%' must be a real,
# unescaped LaTeX comment character, not a literal '\%'.
#   "...width=0.6\linewidth... % fixed: 0.6"        -> exempts "0.6" only
#   "...grid $[0.55, 1.95]$... % fixed: 0.55, 1.95"  -> exempts both
#   "...% fixed: 0.6 -- figure width"                -> free prose after the
#                                                        value list is fine
FIXED_MARKER = re.compile(
    r"(?<!\\)%\s*fixed:\s*(?P<values>(?:[+-]?\d+\.\d+\s*,\s*)*[+-]?\d+\.\d+)"
)

# The \input{generated/...} macro itself: stripped from (not skipping) the
# line it is on (Mn2), so a number that merely shares the line still gets
# linted.
GENERATED_INPUT = re.compile(r"\\input\{\s*generated/[^}]*\}")


def _fixed_exemptions(line: str) -> tuple[str, set[str]]:
    """Split a line into (code_part, exempt_values) using its '% fixed: ...'
    marker, if it has one that actually lists values. code_part is
    everything before the marker (or the whole line, with only
    \\input{generated/...} stripped, if there is no marker); exempt_values is
    the set of literal value strings the marker names -- matched as exact
    strings against what NUMBER_PATTERN/SCI_PATTERN find in code_part, not
    parsed as floats, so "0.60" would not accidentally exempt a written
    "0.6" or vice versa."""
    m = FIXED_MARKER.search(line)
    if not m:
        return GENERATED_INPUT.sub("", line), set()
    code_part = GENERATED_INPUT.sub("", line[: m.start()])
    exempt = {v.strip() for v in m.group("values").split(",")}
    return code_part, exempt


def _manuscript_tex_files(manuscript_root: Path, generated_dir: Path) -> list[Path]:
    """Every .tex file under manuscript_root, recursively, except files under
    generated_dir (a fragment library full of legitimate decimals, not
    manuscript prose) (Mn3) and except anything inside a hidden directory.

    Hidden directories are tool state, not manuscript: /papermentor writes a
    flattened copy of the paper to paper/.papermentor/.cache/merged.tex with
    every include already expanded, so the generated tables land there as literal
    decimals and the lint fires on content it has already cleared in place."""
    return sorted(p for p in manuscript_root.rglob("*.tex")
                  if generated_dir not in p.parents
                  and not any(part.startswith(".") for part in p.relative_to(manuscript_root).parts))


def run_checks(manuscript_root: Path, generated_dir: Path, build_fn=None,
               generated_files: tuple[str, ...] = GENERATED_FILES) -> bool:
    """Run both gate checks against manuscript_root (top-level .tex files to
    lint, searched recursively) and generated_dir (compared against a fresh
    regeneration). Prints every finding; returns True iff both checks pass."""
    ok = True
    build_fn = build if build_fn is None else build_fn

    with tempfile.TemporaryDirectory() as tmp:
        build_fn(Path(tmp))
        for name in generated_files:
            fresh = (Path(tmp) / name).read_text()
            committed_path = generated_dir / name
            if not committed_path.exists():
                ok = False
                print(f"STALE: {generated_dir.name}/{name} does not exist")
                continue
            if fresh != committed_path.read_text():
                ok = False
                print(f"STALE: {generated_dir.name}/{name} differs from a fresh regeneration")

    for tex_path in _manuscript_tex_files(manuscript_root, generated_dir):
        rel = tex_path.relative_to(manuscript_root)
        for i, line in enumerate(tex_path.read_text().splitlines(), 1):
            if line.strip().startswith("%"):
                continue
            code_part, exempt = _fixed_exemptions(line)
            found = NUMBER_PATTERN.findall(code_part) + SCI_PATTERN.findall(code_part)
            if any(value not in exempt for value in found):
                ok = False
                print(f"HARD-CODED NUMBER {rel}:{i}: {line.strip()[:90]}")

    return ok


if __name__ == "__main__":
    all_ok = True
    for root, gen, build_fn, files in TARGETS:
        ok = run_checks(root, gen, build_fn, files)
        print(f"{root.name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)
