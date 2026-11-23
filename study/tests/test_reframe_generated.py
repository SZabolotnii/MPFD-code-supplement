"""The committed macros and tables of the contamination paper (paper/reframe/generated) must be
exactly what study/reframe/make_tex_reframe.py builds from the stored results."""
import filecmp
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reframe"))
import make_tex_reframe  # noqa: E402


def test_reframe_generated_is_current(tmp_path):
    make_tex_reframe.build(tmp_path, tmp_path / "claims.yaml")
    committed = STUDY.parent / "paper" / "reframe" / "generated"
    built = sorted(tmp_path.glob("*.tex"))
    assert built
    for f in built:
        assert filecmp.cmp(f, committed / f.name, shallow=False), f.name
