import numpy as np
import pytest

from mcp import basis as B
from mcp import laws
from mcp import projection as P
from mcp.moments import OneSidedMoments


@pytest.fixture(scope="module")
def pair():
    m = laws.aepd(0.30, 1.2, 3.0)
    return OneSidedMoments(m), OneSidedMoments(laws.affine(m, shift=0.3))


def test_population_matches_sample(pair):
    om, on = pair
    cols = B.halfline([0.8], [1.4])
    cm = P.population_moments(cols, om)
    x = om.law.rvs(2_000_000, np.random.default_rng(7))
    cs = P.sample_moments(cols, x)
    assert np.allclose(cm.mu, cs.mu, rtol=5e-3, atol=2e-3)
    assert np.allclose(cm.C, cs.C, rtol=1e-2, atol=2e-3)


def test_halfline_cross_block_is_rank_one(pair):
    om, _ = pair
    cm = P.population_moments(B.halfline([0.8], [1.4]), om)
    assert abs(cm.C[1, 2] + om.nu(0.8, +1) * om.nu(1.4, -1)) < 1e-12


def test_mixed_equals_halfline_same_exponent(pair):
    om, on = pair
    dm = P.deflection(P.population_moments(B.mixed([0.9]), om), P.population_moments(B.mixed([0.9]), on))
    dh = P.deflection(P.population_moments(B.halfline([0.9], [0.9]), om), P.population_moments(B.halfline([0.9], [0.9]), on))
    assert abs(dm.J - dh.J) < 1e-10
    assert dm.cond < P.COND_LIMIT and dh.cond < P.COND_LIMIT


def test_explicit_s3_formula_matches_solver(pair):
    om, on = pair
    cols = B.halfline([0.8], [1.4])
    j = P.deflection(P.population_moments(cols, om), P.population_moments(cols, on)).J
    assert abs(P.j_halfline_s3(om, on, 0.8, 1.4) - j) < 1e-10


def test_schur_gain_identity(pair):
    om, on = pair
    cols = B.mixed([2.0])                       # x, |x|^2, sgn(x)x^2
    cm, cn = P.population_moments(cols, om), P.population_moments(cols, on)
    j_full = P.deflection(cm, cn).J
    j_head, yt, s = P.schur_gain(cm, cn)
    j_classical = P.deflection(P.population_moments(B.classical(2), om), P.population_moments(B.classical(2), on)).J
    assert abs(j_head - j_classical) < 1e-10
    assert abs(j_full - (j_head + yt ** 2 / s)) < 1e-10 and s > 0 and j_full >= j_classical


def test_parity_split_symmetric_pair():
    om = OneSidedMoments(laws.laplace())
    on = OneSidedMoments(laws.affine(laws.laplace(), scale=1.3))
    cols = B.mixed([0.6, 1.2, 1.4])
    cm, cn = P.population_moments(cols, om), P.population_moments(cols, on)
    r = P.parity_split(cols, cm, cn)
    assert abs(r["J_full"] - r["J_even"]) < 1e-10 and r["J_odd"] < 1e-10 and r["J_even"] > 0.01
    assert P.deflection(cm, cn).cond < P.COND_LIMIT


def test_ceiling_monotone_gaussian_scale():
    lm, ln = laws.gaussian(), laws.affine(laws.gaussian(), scale=1.5)
    om, on = OneSidedMoments(lm), OneSidedMoments(ln)
    delta, jinf = P.j_infinity(lm, ln)
    assert 0 < delta < 2 and jinf > 0
    js = [P.deflection(P.population_moments(B.classical(s), om), P.population_moments(B.classical(s), on)).J
          for s in (1, 2, 3, 4)]
    assert js[0] < 1e-12 and all(js[i] <= js[i + 1] + 1e-12 for i in range(3)) and js[-1] <= jinf


def test_singular_dictionary_reports_nan_and_huge_cond():
    om = OneSidedMoments(laws.laplace())
    on = OneSidedMoments(laws.affine(laws.laplace(), scale=1.3))
    cols = B.mixed([0.6, 1.0, 1.4])          # sgn(x)|x|^1 duplicates the linear column
    d = P.deflection(P.population_moments(cols, om), P.population_moments(cols, on))
    assert d.cond > P.COND_LIMIT
    assert not np.isfinite(d.J) or d.cond > P.COND_LIMIT
