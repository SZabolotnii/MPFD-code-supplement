import numpy as np
import pytest

from mcp import basis as B


def test_exponent_map_anchors_and_inversion():
    for i in (2, 3, 4):
        assert abs(B.p_i(i, 0.0) - 1 / i) < 1e-12
        assert abs(B.p_i(i, 0.5) - 1.0) < 1e-12
        assert abs(B.p_i(i, 1.0) - i) < 1e-12
    assert B.p_i(3, 0.3) < B.p_i(2, 0.3) < 1.0      # ladder inverted below the midpoint
    assert B.p_i(3, 0.8) > B.p_i(2, 0.8) > 1.0


def test_columns_evaluate_and_coefficients():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.allclose(B.Column("abs", 0.5)(x), np.abs(x) ** 0.5)
    assert np.allclose(B.Column("signed", 2.0)(x), np.sign(x) * x ** 2)
    assert np.allclose(B.Column("half_plus", 1.0)(x), np.maximum(x, 0))
    assert np.allclose(B.Column("half_minus", 1.0)(x), np.maximum(-x, 0))
    assert B.Column("linear", 1.0).coef == (1.0, -1.0) and B.Column("abs", 0.7).coef == (1.0, 1.0)
    assert B.Column("half_minus", 0.7).coef == (0.0, 1.0)
    assert B.Column("signed", 0.7).parity == "odd" and B.Column("abs", 0.7).parity == "even"
    assert B.Column("half_plus", 0.7).parity == "none"


def test_dictionaries_have_declared_budget():
    assert len(B.patp(0.3, 3)) == 3 and all(c.kind in ("linear", "signed") for c in B.patp(0.3, 3))
    assert len(B.mixed([0.75])) == 3 and [c.kind for c in B.mixed([0.75])] == ["linear", "abs", "signed"]
    assert len(B.halfline([0.8], [1.4])) == 3
    assert [c.kind for c in B.classical(3)] == ["linear", "abs", "signed"] and B.classical(3)[1].a == 2.0
    assert [c.a for c in B.free_signed([0.4, 1.7])] == [1.0, 0.4, 1.7]


def test_halfline_equals_parity_split_pointwise():
    x = np.linspace(-3, 3, 101)
    a = 0.75
    lhs = B.Column("half_plus", a)(x)
    rhs = 0.5 * (B.Column("abs", a)(x) + B.Column("signed", a)(x))
    assert np.allclose(lhs, rhs)


def test_features_shape_and_admissibility():
    x = np.zeros((7, 5))
    assert B.features(B.mixed([0.5, 1.5]), x).shape == (7, 5, 5)
    assert B.admissible(B.mixed([0.75]), 3.0) and not B.admissible(B.classical(3), 3.0)
    assert not B.admissible(B.mixed([0.75]), 2.0)      # the linear column (a = 1) sits on the boundary at alpha = 2
    assert B.exponent_ceiling(3.0) == 1.5 and B.max_exponent(B.classical(3)) == 3.0
