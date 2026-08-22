import numpy as np
import pytest
from scipy import stats

from mcp import laws


def _mean_var(law):
    return law.expect(lambda x: x), law.expect(lambda x: x * x)


@pytest.mark.parametrize("law", [laws.gaussian(), laws.laplace(), laws.student_t(6),
                                 laws.aepd(0.30, 1.2, 3.0), laws.aepd(0.35, 0.8, 4.0),
                                 laws.fleishman(1.2, 2.0), laws.gauss_mixture([.5, .5], [0, 0], [.6, 1.3])])
def test_standardized(law):
    m, v = _mean_var(law)
    assert abs(m) < 1e-9 and abs(v - 1.0) < 1e-8


@pytest.mark.parametrize("law", [laws.laplace(), laws.aepd(0.30, 1.2, 3.0), laws.fleishman(1.2, 2.0),
                                 laws.contaminated(laws.laplace(), 0.1)])
def test_sampler_matches_expectation(law):
    rng = np.random.default_rng(0)
    x = law.rvs(400_000, rng)
    for f in (lambda t: np.tanh(t), lambda t: np.minimum(np.abs(t), 3.0) ** 0.7):
        mc = f(x).mean()
        se = f(x).std() / np.sqrt(x.size)
        assert abs(mc - law.expect(f)) < 4 * se + 1e-4


def test_aepd_samples_follow_pdf():
    law = laws.aepd(0.35, 0.8, 4.0)
    x = law.rvs(50_000, np.random.default_rng(1))
    grid = np.linspace(-6, 6, 2001)
    cdf = np.cumsum(law.pdf(grid)) * (grid[1] - grid[0])
    p = stats.kstest(x, lambda t: np.interp(t, grid, cdf)).pvalue
    assert p > 0.001


def test_fleishman_moments():
    law = laws.fleishman(1.2, 2.0)
    g3 = law.expect(lambda x: x ** 3)
    g4 = law.expect(lambda x: x ** 4) - 3.0
    assert abs(g3 - 1.2) < 1e-8 and abs(g4 - 2.0) < 1e-7
    assert not law.symmetric and laws.fleishman(0.0, 4.0).symmetric


def test_affine_reflect_contaminated_flags():
    base = laws.laplace()
    s = laws.affine(base, shift=0.3)
    assert abs(s.expect(lambda x: x) - 0.3) < 1e-9 and not s.symmetric
    r = laws.reflect(laws.aepd(0.30, 1.2, 3.0))
    assert abs(r.expect(lambda x: x ** 3) + laws.aepd(0.30, 1.2, 3.0).expect(lambda x: x ** 3)) < 1e-9
    c = laws.contaminated(base, 0.1, df=2)
    assert c.tail_index == 2 and c.symmetric
    assert laws.contaminated(base, 0.0) is base
    assert abs(c.expect(lambda x: np.ones_like(x)) - 1.0) < 1e-9


def test_heavy_tail_moment_near_tail_index():
    from scipy.special import gamma
    law = laws.student_t(2)                      # unit scale, tail index 2
    q = 1.9
    closed = 2 ** (q / 2) * gamma((q + 1) / 2) * gamma((2 - q) / 2) / (np.sqrt(np.pi) * gamma(1.0))
    assert abs(law.expect(lambda x: np.abs(x) ** q) - closed) / closed < 1e-6
