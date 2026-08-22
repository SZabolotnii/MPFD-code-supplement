import numpy as np
import pytest

from mcp import laws
from mcp.moments import (InadmissibleMoment, OneSidedMoments, SampleOneSidedMoments,
                         gaussian_nu_plus, hill_tail_index, laplace_nu_plus, student_t_nu_plus)


@pytest.mark.parametrize("q", [0.3, 0.75, 1.0, 1.6, 2.0, 2.9])
def test_closed_forms(q):
    assert abs(OneSidedMoments(laws.gaussian()).nu(q, +1) - gaussian_nu_plus(q)) < 1e-10
    assert abs(OneSidedMoments(laws.laplace()).nu(q, -1) - laplace_nu_plus(q)) < 1e-10
    assert abs(OneSidedMoments(laws.student_t(6)).nu(q, +1) - student_t_nu_plus(q, 6)) < 1e-9


def test_symmetric_law_has_zero_sigma():
    o = OneSidedMoments(laws.laplace())
    assert abs(o.sigma(0.7)) < 1e-12 and abs(o.sigma(1.9)) < 1e-12
    assert abs(o.nu_total(2.0) - 1.0) < 1e-9


def test_inadmissible_raises():
    o = OneSidedMoments(laws.contaminated(laws.laplace(), 0.1, df=2))
    assert o.tail_index == 2
    assert o.nu(0.75, +1) > 0
    with pytest.raises(InadmissibleMoment):
        o.nu(2.0, +1)
    with pytest.raises(InadmissibleMoment):
        o.nu(3.0, -1)


def test_sample_moments_converge():
    law = laws.aepd(0.30, 1.2, 3.0)
    pop = OneSidedMoments(law)
    x = law.rvs(2_000_000, np.random.default_rng(3))
    smp = SampleOneSidedMoments(x)
    for q in (0.5, 1.0, 1.5):
        for side in (+1, -1):
            assert abs(smp.nu(q, side) - pop.nu(q, side)) / pop.nu(q, side) < 5e-3


def test_hill_recovers_t_index():
    rng = np.random.default_rng(5)
    x = rng.standard_t(2, 400_000)
    est = hill_tail_index(x, frac=0.02)
    assert 1.6 < est < 2.5
