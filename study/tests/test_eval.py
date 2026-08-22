import numpy as np

from mcp import basis as B
from mcp import eval as E
from mcp import panel


def test_auc_basic():
    assert E.auc(np.array([0., 1., 2.]), np.array([3., 4.])) == 1.0
    assert abs(E.auc(np.array([0., 2.]), np.array([1., 3.])) - 0.75) < 1e-12
    assert E.auc(np.array([3., 4.]), np.array([0., 1.])) == 1.0   # orientation-free


def test_window_scores_and_bootstrap():
    rng = np.random.default_rng(0)
    w1 = rng.normal(0, 1, (500, 64))
    w2 = rng.normal(0.3, 1, (500, 64))
    mu, sd = E.standardizer(w1, w2)
    cols = B.classical(2)
    K = np.array([1.0, 0.0])
    s1, s2 = E.window_scores(cols, K, w1, mu, sd), E.window_scores(cols, K, w2, mu, sd)
    a = E.auc(s1, s2)
    lo, hi = E.paired_window_bootstrap(s1, s2, 100, rng)
    assert 0.9 < a <= 1.0 and lo <= a <= hi
    lo2, hi2 = E.moving_block_bootstrap(s1, s2, 8, 100, rng)
    assert lo2 <= a <= hi2


def test_panel_pairs_exist():
    for name, (m, n) in {**panel.lp1(), **panel.lp2(), **panel.lp3(), **panel.lp4()}.items():
        assert m.pdf is not None and n.pdf is not None, name
    m, n = panel.lp5(0.1)
    assert m.tail_index == 2 and n.tail_index == 2
    assert panel.lp5(0.1, df=3)[0].tail_index == 3
    m0, n0 = panel.lp5(0.0)
    assert m0.symmetric and not n0.symmetric
