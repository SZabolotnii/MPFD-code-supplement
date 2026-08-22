"""Pre-registered class pairs LP1-LP5 (CHECKS.md, Letter spec section 3.1)."""
from __future__ import annotations

from . import laws
from .laws import Law

A2L = laws.aepd(0.30, 1.2, 3.0, name="A2L")
A4L = laws.aepd(0.35, 0.8, 4.0, name="A4L")
A5 = laws.aepd(0.25, 0.7, 3.5, name="A5")
A1L = laws.aepd(0.30, 2.0, 2.0, name="A1L")
A1R = laws.aepd(0.70, 2.0, 2.0, name="A1R")
SHIFT = 0.3
MEAN_SHIFT = 0.05


def lp1() -> dict[str, tuple[Law, Law]]:
    l = laws.laplace()
    return {"LP1 laplace scale 1 vs 1.3": (l, laws.affine(l, scale=1.3))}


def lp2() -> dict[str, tuple[Law, Law]]:
    a = laws.gauss_mixture([0.5, 0.5], [0.0, 0.0], [0.6, 1.3])   # gamma4 ~ 1.26
    b = laws.gauss_mixture([0.5, 0.5], [0.0, 0.0], [0.3, 1.6])   # gamma4 ~ 2.61
    return {"LP2 symmetric mixtures g4 1.26 vs 2.61": (a, b)}


def lp3() -> dict[str, tuple[Law, Law]]:
    return {f"LP3 {law.name} shift {SHIFT}": (law, laws.affine(law, shift=SHIFT)) for law in (A2L, A4L, A5)}


def lp4() -> dict[str, tuple[Law, Law]]:
    return {"LP4 A1L vs A1R": (A1L, A1R), "LP4 A2L vs reflect(A2L)": (A2L, laws.reflect(A2L))}


def lp5(eps: float, df: float = 2) -> tuple[Law, Law]:
    drone = laws.fleishman(0.0, 4.0)
    pult = laws.affine(laws.fleishman(1.2, 2.0), shift=MEAN_SHIFT)
    return laws.contaminated(drone, eps, df=df), laws.contaminated(pult, eps, df=df)


# ----------------------------------------------------------------- TSP panel (CHECKS.md, TSP section)
def tp1() -> tuple[Law, Law]:
    """Gaussian vs Laplace: the first two moments agree, chi^2(Laplace||Gaussian) = infinity."""
    return laws.gaussian(), laws.laplace()


def tp2(delta: float = 0.5) -> tuple[Law, Law]:
    l = laws.laplace()
    return l, laws.affine(l, shift=delta)


def tp3() -> dict[str, tuple[Law, Law]]:
    return lp3()


def tp4() -> tuple[Law, Law]:
    return lp5(0.0)


def tp5(delta: float = 0.5) -> dict[str, tuple[Law, Law]]:
    out = {}
    for name, law in (("t(3)", laws.student_t(3)), ("t(2)", laws.student_t(2)), ("cauchy", laws.student_t(1))):
        out[f"TP5 {name} shift {delta}"] = (law, laws.affine(law, shift=delta))
    return out


def tsp_pairs() -> dict[str, tuple[Law, Law]]:
    """Named binary pairs of the TSP checks (TP6 is the M-ary SPL panel, see scenarios.py)."""
    out = {"TP1 gaussian vs laplace": tp1(), "TP2 laplace shift 0.5": tp2(0.5)}
    out.update({k.replace("LP3", "TP3"): v for k, v in tp3().items()})
    out["TP4 drone vs pult"] = tp4()
    out.update(tp5())
    return out
