"""Bridge to the sibling Ku-HL-PATP engine: same nu^+/- on shared laws (no code copied).

HL truncates quadrature at ±200; the bridge corrects for that tail (≈2.7e-7 at q=2.6 on t(6)).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from mcp import laws
from mcp.moments import OneSidedMoments

HL = Path("/Users/docua/Project/Research/Ku-HL-PATP")
if not (HL / "scripts" / "hl_patp_core.py").exists():
    pytest.skip("Ku-HL-PATP not available on this host", allow_module_level=True)
sys.path.insert(0, str(HL))
sys.path.insert(0, str(HL / "scripts"))
hl = pytest.importorskip("hl_patp_core")


def _tail_beyond_200(law, q, side):
    if side > 0:
        return law.expect(lambda x: np.where(x > 200.0, np.abs(x) ** q, 0.0))
    return law.expect(lambda x: np.where(x < -200.0, np.abs(x) ** q, 0.0))


@pytest.mark.parametrize("ours,theirs", [
    (laws.laplace(), hl.reference_laws()["laplace"]),
    (laws.student_t(6), hl.reference_laws()["student_t6"]),
    (laws.aepd(0.30, 1.2, 3.0), hl.standardized_aepd(0.30, 1.2, 3.0)),
    (laws.aepd(0.35, 0.8, 4.0), hl.standardized_aepd(0.35, 0.8, 4.0)),
])
@pytest.mark.parametrize("q", [0.55, 1.0, 1.45, 1.95, 2.6])
def test_nu_matches_hl(ours, theirs, q):
    a = OneSidedMoments(ours)
    b = hl.OneSidedMoments(theirs)
    for side in (+1, -1):
        # HL integrates on [-200, 200]; add back what it cannot see so the comparison is like with like
        assert abs(a.nu(q, side) - _tail_beyond_200(ours, q, side) - b.nu(q, side)) < 1e-8
