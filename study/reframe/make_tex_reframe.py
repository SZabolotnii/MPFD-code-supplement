"""Generate paper/reframe/generated/ (numbers.tex, tables) and paper/reframe/claims.yaml.

Every result number of the reframe manuscript is a macro defined here from the hashed
artifacts of the two runs (results/h1v4-2026-11-22/, results/h1v4-s2-2026-11-22/) and the two
diagnostics (results/diag_*.json). Nothing is recomputed: values are read and aggregated
(min / max / median over cells). The claim ledger records, for each macro, the artifacts and
the aggregation, so /coe-audit can trace a number in the PDF back to a file.

Usage: python make_tex_reframe.py   (writes into the committed locations)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
R1 = RES / "h1v4-2026-11-22"
R2 = RES / "h1v4-s2-2026-11-22"
R3 = RES / "h1v4-s3-2026-11-23"
R4 = RES / "h1v4-s4-2026-11-24"
PAPER = HERE.parent.parent / "paper" / "reframe"
OUT = PAPER / "generated"
FILES = ("numbers.tex", "tab_criteria.tex", "tab_fast.tex", "tab_slow.tex", "tab_s2.tex",
         "tab_s3a.tex", "tab_s3b.tex", "tab_s4a.tex", "tab_s4b.tex", "tab_s4c.tex")
DICT_ORDER = ("oo", "ee", "oe", "eo")
DICT_TEX = {"oo": r"$\mathsf{oo}$", "ee": r"$\mathsf{ee}$", "oe": r"$\mathsf{oe}$", "eo": r"$\mathsf{eo}$"}


def _cells(run: Path) -> list[dict]:
    out = []
    for f in sorted(run.glob("*.json")):
        if f.name == "RUN.json":
            continue
        a = json.loads(f.read_text())
        a["file"] = f"study/reframe/results/{run.name}/{f.name}"
        out.append(a)
    return out


def f(x: float, d: int) -> str:
    return f"{x:.{d}f}"


def g(x: float) -> str:
    """Integer-like large number with thin-space grouping (e.g. 16\\,283)."""
    s = f"{int(round(x)):,}"
    return s.replace(",", r"\,")


class Book:
    def __init__(self):
        self.macros: list[str] = []
        self.claims: list[dict] = []

    def add(self, name: str, text: str, value: float, files: list[str], how: str):
        self.macros.append(rf"\newcommand{{\{name}}}{{{text}}}")
        self.claims.append({"macro": name, "raw": text, "value": float(value),
                            "files": sorted(set(files)), "aggregation": how})


def build(out_dir: Path, claims_path: Path | None = None) -> None:
    c1, c2 = _cells(R1), _cells(R2)
    A = [c for c in c1 if c["params"].get("block") == "A"]
    B = {c["params"]["dict"]: c for c in c1 if c["params"].get("block") == "B"}
    X = [c for c in c2 if c["params"].get("block") == "X"]
    T = {c["params"]["dict"]: c for c in c2 if c["params"].get("block") == "T"}
    c3 = _cells(R3)
    SA = [c for c in c3 if c["params"].get("block") == "A"]
    SB = [c for c in c3 if c["params"].get("block") == "B"]
    SC = {c["params"]["dict"]: c for c in c3 if c["params"].get("block") == "C"}
    c4 = _cells(R4)
    U = {b: [c for c in c4 if c["params"].get("block") == b] for b in ("D", "A4", "B4", "C4", "E4")}
    C2 = {c["params"]["ctl"]: c for c in c2 if c["params"].get("block") == "control"}
    d1 = json.loads((RES / "diag_competitor.json").read_text())
    d2 = json.loads((RES / "diag_clean_mean.json").read_text())
    fd1 = "study/reframe/results/diag_competitor.json"
    fd2 = "study/reframe/results/diag_clean_mean.json"
    bk = Book()
    fa = [c["file"] for c in A]

    # ---------------- run 1, Block A
    same = [c for c in A if c["values"]["same_parity"] == 1.0]
    opp = [c for c in A if c["values"]["same_parity"] == 0.0 and c["params"]["pplus"] == 0.5]
    p1_pass = sum(abs(c["values"]["slope_err"]) <= 0.10 for c in A)
    bk.add("rfPonePass", str(p1_pass), p1_pass, fa, "count |slope_err| <= 0.10 over Block A")
    bk.add("rfNcellsA", str(len(A)), len(A), fa, "count Block A cells")
    dmax = max(c["values"]["dauc_n128000"] for c in A)
    bk.add("rfPtwoMax", f(dmax, 4), dmax, fa, "max dauc_n128000 over Block A")
    for n, nm in ((2000, "Two"), (32000, "ThirtyTwo"), (128000, "Max")):
        v = min(c["values"][f"lock_n{n}"] for c in same)
        bk.add(f"rfLockMin{nm}", f(v, 2), v, [c["file"] for c in same], f"min lock_n{n} over same-parity Block A")
    bk.add("rfNsame", str(len(same)), len(same), [c["file"] for c in same], "count same-parity Block A cells")
    p4_pass = sum(abs(c["values"]["pos_n128000"] - 0.5) <= 0.087 for c in opp)
    bk.add("rfPfourPass", str(p4_pass), p4_pass, [c["file"] for c in opp], "count |pos-1/2|<=0.087, opposite parity p+=1/2")
    bk.add("rfNopp", str(len(opp)), len(opp), [c["file"] for c in opp], "count opposite-parity p+=1/2 cells")
    for a_, nm in ((2.4, "Low"), (4.0, "High")):
        oe = [c for c in A if c["params"]["dict"] == "oe" and c["params"]["astar"] == a_
              and c["params"]["pplus"] == 0.5]
        lo, hi = min(c["values"]["pos_n128000"] for c in oe), max(c["values"]["pos_n128000"] for c in oe)
        bk.add(f"rfPosOe{nm}Lo", f(lo, 2), lo, [c["file"] for c in oe], f"min pos_n128000, oe a*={a_}, p+=1/2")
        bk.add(f"rfPosOe{nm}Hi", f(hi, 2), hi, [c["file"] for c in oe], f"max pos_n128000, oe a*={a_}, p+=1/2")
    for a_, nm in ((4.0, "High"), (2.4, "Low")):
        ks = [c["values"]["ks_H"] for c in same if c["params"]["astar"] == a_]
        fs = [c["file"] for c in same if c["params"]["astar"] == a_]
        bk.add(f"rfKs{nm}Lo", f(min(ks), 2), min(ks), fs, f"min ks_H, same parity a*={a_}")
        bk.add(f"rfKs{nm}Hi", f(max(ks), 2), max(ks), fs, f"max ks_H, same parity a*={a_}")
    ks = [c["values"]["ks_H"] for c in opp]
    bk.add("rfKsOppLo", f(min(ks), 2), min(ks), [c["file"] for c in opp], "min ks_H, opposite parity, p+=1/2")
    bk.add("rfKsOppHi", f(max(ks), 2), max(ks), [c["file"] for c in opp], "max ks_H, opposite parity, p+=1/2")
    # p+ = 4/5: the run-1 H sampler used the conditional tail variance 4p+p- instead of the full one;
    # diag_h_tail.py bounds the resulting shift of each ks_H by ks(H_old, H_new) (rfHtailKsMax)
    opp8 = [c for c in A if c["values"]["same_parity"] == 0.0 and c["params"]["pplus"] == 0.8]
    for a_, nm in ((4.0, "High"), (2.4, "Low")):
        cs = [c for c in opp8 if c["params"]["astar"] == a_]
        ks = [c["values"]["ks_H"] for c in cs]
        bk.add(f"rfKsOppEight{nm}Lo", f(min(ks), 2), min(ks), [c["file"] for c in cs],
               f"min ks_H, opposite parity, p+=4/5, a*={a_}")
        bk.add(f"rfKsOppEight{nm}Hi", f(max(ks), 2), max(ks), [c["file"] for c in cs],
               f"max ks_H, opposite parity, p+=4/5, a*={a_}")
    d6 = json.loads((RES / "diag_h_tail.json").read_text())
    ht = max(v["ks_old_new"] for v in d6.values())
    bk.add("rfHtailKsMax", f(ht, 4), ht, ["study/reframe/results/diag_h_tail.json"],
           "max ks(H_old, H_new) over opposite-parity H laws, conditional vs Campbell tail variance")
    # ---------------- run 1, Block B
    fb = [c["file"] for c in B.values()]
    for d in DICT_ORDER:
        v = B[d]["values"]
        bk.add(f"rfOnSlope{d.upper()}", f(v["slope"], 2), v["slope"], [B[d]["file"]], f"Block B onset slope, {d}")
        bk.add(f"rfRatio{d.upper()}", f(v["ratio_n1000"], 3), v["ratio_n1000"], [B[d]["file"]], f"Block B ratio_n1000, {d}")
    s2e = max(B[d]["values"][f"S2err_n{n}"] for d in B for n in (50, 200, 1000))
    bk.add("rfPeightMax", f(s2e, 4), s2e, fb, "max S2err over Block B cells and n")
    for n, nm in ((50, "Fifty"), (1000, "Thousand")):
        vals = [B[d]["values"][f"medNon_n{n}"] for d in B]
        bk.add(f"rfNonB{nm}Lo", sci(min(vals)), min(vals), fb, f"min medNon_n{n} over Block B")
        bk.add(f"rfNonB{nm}Hi", sci(max(vals)), max(vals), fb, f"max medNon_n{n} over Block B")
    pk = [B[d]["values"][f"medNpeak_n{n}"] for d in B for n in (50, 200, 1000)]
    bk.add("rfNpeakLo", g(min(pk)), min(pk), fb, "min medNpeak over Block B")
    bk.add("rfNpeakHi", g(max(pk)), max(pk), fb, "max medNpeak over Block B")
    # ---------------- diagnostics
    bk.add("rfDoneFullFour", f(d1["oe_a4.0_bal"]["pos_full"], 2), d1["oe_a4.0_bal"]["pos_full"], [fd1], "oe a*=4 bal pos_full")
    bk.add("rfDoneRedFour", f(d1["oe_a4.0_bal"]["pos_reduced"], 2), d1["oe_a4.0_bal"]["pos_reduced"], [fd1], "oe a*=4 bal pos_reduced")
    bk.add("rfDoneFullLow", f(d1["oe_a2.4_bal"]["pos_full"], 2), d1["oe_a2.4_bal"]["pos_full"], [fd1], "oe a*=2.4 bal pos_full")
    bk.add("rfDoneRedLow", f(d1["oe_a2.4_bal"]["pos_reduced"], 2), d1["oe_a2.4_bal"]["pos_reduced"], [fd1], "oe a*=2.4 bal pos_reduced")
    for k, nm in (("bal", "Bal"), ("plus", "Plus"), ("minus", "Minus")):
        for a_, an in (("4.0", "Four"), ("2.4", "Low")):
            v = d2[f"oe_reduced_eqscale_a{a_}_{k}"]["pos"]
            bk.add(f"rfDtwo{nm}{an}", f(v, 2 if k != "bal" or a_ != "4.0" else 3), v, [fd2], f"eqscale reduced oe a*={a_} {k} pos")
    v = d2["oe_reduced_eqscale_a4.0_bal"]["clean_Ydiff_registered_laws"]
    bk.add("rfCleanYdiff", f(v, 2), v, [fd2], "clean_Ydiff_registered_laws, a*=4")
    v = d2["oe_reduced_eqscale_a4.0_bal"]["clean_Ydiff_equal_scale"]
    bk.add("rfCleanYdiffEq", f(v, 2), v, [fd2], "clean_Ydiff_equal_scale, a*=4")
    # ---------------- post-hoc diagnostic of the Q1-failing cells (diag_s2_onset_censoring.py)
    d3 = json.loads((RES / "diag_s2_onset_censoring.json").read_text())
    fd3 = "study/reframe/results/diag_s2_onset_censoring.json"
    low = [d3[k] for k in d3 if "n=50_" in k]
    for key, nm, fmt in (("frac_low_peak", "CensLow", 2), ("medNstar_low_peak", "CensNstarLow", 0),
                         ("medNstar_finite", "CensNstarFin", 0)):
        xs = [c[key] for c in low]
        for agg, an in ((min, "Lo"), (max, "Hi")):
            v = agg(xs)
            bk.add(f"rf{nm}{an}", g(v) if fmt == 0 else f(v, fmt), v, [fd3], f"{agg.__name__} {key}, Q1-failing n=50 cells")
    d4 = json.loads((RES / "diag_b_clean_drift.json").read_text())
    fd4 = "study/reframe/results/diag_b_clean_drift.json"
    rows = [(d, k, v) for d, r in d4.items() for k, v in r.items() if k.startswith("n")]
    diffs = [abs(v["med_ratio_eff"] - v["stored_onset_ratio"]) for _, _, v in rows]
    match = sum(x <= 0.02 for x in diffs)
    bk.add("rfDriftMatch", str(match), match, [fd4], "count dict x n with |med N*_eff/N* - stored ratio| <= 0.02")
    drop = max(40 - v[f"n{n}"]["n_ok"] for v in d4.values() for n in (50, 200, 1000))
    bk.add("rfDriftDropMax", str(drop), drop, [fd4], "max replications (of 40) with mu_eff <= 0, left out")
    bk.add("rfDriftCells", str(len(rows)), len(rows), [fd4], "count dict x n cells, Block B")
    v = d4["oe"]["n1000"]["med_ratio_eff"]
    bk.add("rfDriftOe", f(v, 3), v, [fd4], "oe n=1000 median N*_eff/N*")
    worst = max(rows, key=lambda r: abs(r[2]["med_ratio_eff"] - r[2]["stored_onset_ratio"]))
    bk.add("rfDriftWorstEff", f(worst[2]["med_ratio_eff"], 2), worst[2]["med_ratio_eff"], [fd4], f"worst cell {worst[0]} {worst[1]} med N*_eff/N*")
    bk.add("rfDriftWorstObs", f(worst[2]["stored_onset_ratio"], 2), worst[2]["stored_onset_ratio"], [fd4], f"worst cell {worst[0]} {worst[1]} stored onset ratio")
    assert (worst[0], worst[1]) == ("ee", "n50"), worst[:2]
    d5 = json.loads((RES / "diag_s2_ksub.json").read_text())
    fd5 = "study/reframe/results/diag_s2_ksub.json"
    for key, nm in (("rfly_oe", "RflyOe"), ("gauss_oe", "GaussOe")):
        v = d5[key]["rel_lstar"]
        bk.add(f"rfKsubRel{nm}", f(v, 3 if v < 0.01 else 2), v, [fd5], f"{key} |Ksub_l*| / max|Ksub|")
    v = d5["rfly_oe"]["Ksub_lstar"]
    bk.add("rfKsubLstarRflyOe", f(v, 4), v, [fd5], "rfly_oe Ksub_l*")
    ry = d3["dict=oe_n=1000_src=rfly_te=sym"]
    bk.add("rfCensRflyNoDrop", f(ry["frac_no_drop"], 2), ry["frac_no_drop"], [fd3], "rfly oe n=1000 frac_no_drop")
    bk.add("rfCensRflyNstarNoDrop", sci(ry["medNstar_no_drop"]), ry["medNstar_no_drop"], [fd3], "rfly oe n=1000 medNstar_no_drop")
    # ---------------- S2
    fx = [c["file"] for c in X]
    sym = [c for c in X if c["params"]["te"] == "sym"]
    one = [c for c in X if c["params"]["te"] == "one"]
    synt = [c for c in sym if c["params"]["src"] == "gauss"]
    fs, fo, fy = [c["file"] for c in sym], [c["file"] for c in one], [c["file"] for c in synt]
    vals = lambda cs, k: [c["values"][k] for c in cs]
    bk.add("rfSPeakLo", f(min(vals(sym, "medPeak")), 2), min(vals(sym, "medPeak")), fs, "min medPeak, symmetric S2")
    bk.add("rfSPeakHi", f(max(vals(sym, "medPeak")), 2), max(vals(sym, "medPeak")), fs, "max medPeak, symmetric S2")
    bk.add("rfSNonLo", g(min(vals(sym, "medNon"))), min(vals(sym, "medNon")), fs, "min medNon, symmetric S2")
    bk.add("rfSNonHi", g(max(vals(sym, "medNon"))), max(vals(sym, "medNon")), fs, "max medNon, symmetric S2")
    bk.add("rfSRatioLo", f(min(vals(synt, "ratio")), 2), min(vals(synt, "ratio")), fy, "min ratio, symmetric synthetic + t3")
    bk.add("rfSRatioHi", f(max(vals(synt, "ratio")), 2), max(vals(synt, "ratio")), fy, "max ratio, symmetric synthetic + t3")
    bk.add("rfSNsymSynt", str(len(synt)), len(synt), fy, "count symmetric synthetic + t3 cells")
    bk.add("rfSNsym", str(len(sym)), len(sym), fs, "count symmetric S2 cells")
    bk.add("rfSNone", str(len(one)), len(one), fo, "count one-sided S2 cells")
    q1 = sum(c["values"]["medPeak"] >= 0.80 and c["values"]["onFinite"] >= 0.80 and c["values"]["medNon"] <= 2e4 for c in sym)
    bk.add("rfSQonePass", str(q1), q1, fs, "count Q1 pass")
    bk.add("rfSPredErrMax", f(max(vals(one, "predErr")), 3), max(vals(one, "predErr")), fo, "max predErr, one-sided")
    bk.add("rfSWendMin", f(min(vals(sym, "medAucWEnd")), 3), min(vals(sym, "medAucWEnd")), fs, "min medAucWEnd, symmetric")
    bk.add("rfSWendOneMin", f(min(vals(one, "medAucWEnd")), 3), min(vals(one, "medAucWEnd")), fo,
           "min medAucWEnd, one-sided")
    bk.add("rfSCostLo", f(min(vals(sym, "medCostW10")), 3), min(vals(sym, "medCostW10")), fs, "min medCostW10, symmetric")
    bk.add("rfSCostHi", f(max(vals(sym, "medCostW10")), 3), max(vals(sym, "medCostW10")), fs, "max medCostW10, symmetric")
    bk.add("rfSSubEndMin", f(min(vals(sym, "medAucSEnd")), 3), min(vals(sym, "medAucSEnd")), fs, "min medAucSEnd, symmetric")
    bk.add("rfSOnsetFactor", f(sym[0]["values"]["lambda_m_pow"], 3), sym[0]["values"]["lambda_m_pow"], fs, "lambda_m_pow (identical over cells)")
    lk = [c for c in X if c["params"]["src"] == "gauss" and c["params"]["cont"] == "pareto" and c["params"]["n"] <= 200
          and c["params"]["dict"] in ("oo", "ee")]
    bk.add("rfSLockSmallLo", f(min(vals(lk, "lock")), 2), min(vals(lk, "lock")), [c["file"] for c in lk], "min lock, synthetic same parity n<=200")
    bk.add("rfSLockSmallHi", f(max(vals(lk, "lock")), 2), max(vals(lk, "lock")), [c["file"] for c in lk], "max lock, synthetic same parity n<=200")
    lk1 = [c for c in X if c["params"]["src"] == "gauss" and c["params"]["cont"] == "pareto" and c["params"]["n"] == 1000
           and c["params"]["dict"] in ("oo", "ee")]
    bk.add("rfSLockThousandLo", f(min(vals(lk1, "lock")), 2), min(vals(lk1, "lock")), [c["file"] for c in lk1], "min lock, synthetic same parity n=1000")
    bk.add("rfSLockThousandHi", f(max(vals(lk1, "lock")), 2), max(vals(lk1, "lock")), [c["file"] for c in lk1], "max lock, synthetic same parity n=1000")
    oe1 = [c for c in X if c["params"]["dict"] == "oe" and c["params"]["te"] == "one" and c["params"]["cont"] == "pareto"]
    bk.add("rfOeOnePosLo", f(min(vals(oe1, "pos")), 2), min(vals(oe1, "pos")), [c["file"] for c in oe1], "min share Kt>0, oe one-sided pareto")
    bk.add("rfOeOnePosHi", f(max(vals(oe1, "pos")), 2), max(vals(oe1, "pos")), [c["file"] for c in oe1], "max share Kt>0, oe one-sided pareto")
    bk.add("rfOeOneCells", str(len(oe1)), len(oe1), [c["file"] for c in oe1], "count oe one-sided pareto cells")
    rf = {(c["params"]["dict"], c["params"]["n"], c["params"]["te"]): c for c in X if c["params"]["src"] == "rfly"}
    c_ = rf[("ee", 200, "one")]
    bk.add("rfRflyInvTwo", f(c_["values"]["medAucEnd"], 3), c_["values"]["medAucEnd"], [c_["file"]], "rfly ee n=200 one medAucEnd")
    c_ = rf[("ee", 1000, "one")]
    bk.add("rfRflyInvThousand", f(c_["values"]["medAucEnd"], 3), c_["values"]["medAucEnd"], [c_["file"]], "rfly ee n=1000 one medAucEnd")
    c_ = rf[("ee", 200, "sym")]
    bk.add("rfRflyLockTwo", f(c_["values"]["lock"], 2), c_["values"]["lock"], [c_["file"]], "rfly ee n=200 lock")
    c_ = rf[("oe", 1000, "sym")]
    bk.add("rfRflyOeOnFinite", f(c_["values"]["onFinite"], 2), c_["values"]["onFinite"], [c_["file"]], "rfly oe n=1000 sym onFinite")
    bk.add("rfRflyOeNstar", sci(c_["values"]["medNstar"]), c_["values"]["medNstar"], [c_["file"]], "rfly oe n=1000 sym medNstar")
    bk.add("rfRflyFlights", g(C2["C8"]["values"]["n_flights"]), C2["C8"]["values"]["n_flights"], [C2["C8"]["file"]], "C8 n_flights")
    bk.add("rfRflySdRatio", f(C2["C8"]["values"]["sd_ratio"], 2), C2["C8"]["values"]["sd_ratio"], [C2["C8"]["file"]], "C8 sd_ratio (training half)")
    ee = T["ee"]["values"]
    bk.add("rfTlockTwoK", f(ee["lock_n2000"], 2), ee["lock_n2000"], [T["ee"]["file"]], "t3 ee lock_n2000")
    bk.add("rfTlockMax", f(ee["lock_n128000"], 2), ee["lock_n128000"], [T["ee"]["file"]], "t3 ee lock_n128000")
    td = max(T[d]["values"][k] for d in T for k in ("dauc_n2000", "dauc_n32000", "dauc_n128000"))
    bk.add("rfTdaucMax", f(td, 4), td, [T[d]["file"] for d in T], "max dauc over t3 training cells")
    bk.add("rfTposOe", f(T["oe"]["values"]["pos_n128000"], 2), T["oe"]["values"]["pos_n128000"], [T["oe"]["file"]], "t3 oe pos_n128000")
    bk.add("rfTcZero", f(C2["C7"]["values"]["c0_exact_x1e3"], 4), C2["C7"]["values"]["c0_exact_x1e3"], [C2["C7"]["file"]], "C7 exact x^3 P(|T|>x) at 1e3")

    # ---------------- general-p+ sign limit of Thm 5(ii) vs run-1 frequencies (diag_signlock_pplus.py)
    d7 = json.loads((RES / "diag_signlock_pplus.json").read_text())
    fd7 = "study/reframe/results/diag_signlock_pplus.json"
    for dct in ("eo", "oe"):
        for a_, nm in ((4.0, "High"), (2.4, "Low")):
            cs = [v for v in d7.values() if v["dict"] == dct and v["astar"] == a_ and v["pplus"] == 0.8]
            pr = sorted({round(v["pred"], 6) for v in cs})
            assert len(pr) == 1, pr
            tag = f"{dct.capitalize()}{nm}"
            bk.add(f"rfSignPred{tag}", f(pr[0], 2), pr[0], [fd7], f"rho prediction P(K_t>0), {dct} a*={a_}, p+=4/5")
            obs = [v["pos_n128000"] for v in cs]
            bk.add(f"rfSignObs{tag}Lo", f(min(obs), 2), min(obs), [fd7], f"min pos_n128000, {dct} a*={a_}, p+=4/5")
            bk.add(f"rfSignObs{tag}Hi", f(max(obs), 2), max(obs), [fd7], f"max pos_n128000, {dct} a*={a_}, p+=4/5")

    # ---------------- 5 % two-sample-free KS critical value for the 300 Block A replications
    # (asymptotic one-sample value 1.358/sqrt(R); R = 300 is the design constant R_A of h1v4.py)
    ra = 300
    kc = 1.358 / ra ** 0.5
    bk.add("rfKsCrit", f(kc, 2), kc, [], "1.358/sqrt(300), asymptotic 5% one-sample KS critical value, R_A = 300")
    # ---------------- K^sub_{l*} of ee on the Gaussian and RflyMAD training mixtures (diag_s2_ksub.py)
    for src, nm in (("gauss", "Gauss"), ("rfly", "Rfly")):
        v = d5[f"{src}_ee"]["Ksub_lstar"]
        bk.add(f"rfKsubLstar{nm}Ee", f(v, 2), v, [fd5], f"Ksub_l* of ee, {src} training mixtures, a* = 8")
    # ---------------- run-2 grid: size and integer end point, as stored with each cell
    grids = {tuple(c["values"]["N_grid"]) for c in X}
    assert len(grids) == 1, len(grids)
    grid = next(iter(grids))
    bk.add("rfNgridS", str(len(grid)), len(grid), [X[0]["file"]], "number of N values in the run-2 grid")
    bk.add("rfNmaxSint", g(grid[-1]), grid[-1], [X[0]["file"]], "last N of the run-2 grid (round(10^4.5))")

    # ---------------- run 3 controls
    C3 = {c["params"]["ctl"]: c for c in c3 if c["params"].get("block") == "control"}
    bk.add("rfCtlSthreeZ", f(C3["C11"]["values"]["maxz"], 2), C3["C11"]["values"]["maxz"], [C3["C11"]["file"]],
           "C11 max |z|, Pareto(alpha) sampler")
    bk.add("rfCtlSthreeErr", f(C3["C12"]["values"]["maxerr"], 4), C3["C12"]["values"]["maxerr"], [C3["C12"]["file"]],
           "C12 max error, stable CDF vs LePage at new gamma")
    bk.add("rfCtlSthreeDiff", f(C3["C13"]["values"]["maxdiff"], 0), C3["C13"]["values"]["maxdiff"], [C3["C13"]["file"]],
           "C13 max difference, S3 engine vs run-2 engine")
    oe_drop = 40 - d4["oe"]["n1000"]["n_ok"]
    bk.add("rfDriftDropOeThousand", str(oe_drop), oe_drop, [fd4], "replications left out, oe n = 1000")

    # ---------------- run 3 (S3, DESIGN_S3.md)
    sa_sym = [c for c in SA if c["params"]["te"] == "sym"]
    sa_one = [c for c in SA if c["params"]["te"] == "one"]
    fs = lambda cs: [c["file"] for c in cs]
    r1 = [c for c in sa_sym if c["values"]["medPeak"] >= 0.80 and c["values"]["onFinite"] >= 0.80
          and c["values"]["medNon"] <= 2e4]
    bk.add("rfRonePass", str(len(r1)), len(r1), fs(sa_sym), "count R1 pass, S3-A symmetric")
    bk.add("rfRoneN", str(len(sa_sym)), len(sa_sym), fs(sa_sym), "count S3-A symmetric cells")
    bk.add("rfRthreeN", str(len(sa_one) + len(SB)), len(sa_one) + len(SB), fs(sa_one + SB), "count R3 cells")
    rat = [c["values"]["ratio"] for c in sa_sym]
    bk.add("rfSthreeRatioLo", f(min(rat), 2), min(rat), fs(sa_sym), "min onset ratio, S3-A symmetric")
    bk.add("rfSthreeRatioHi", f(max(rat), 2), max(rat), fs(sa_sym), "max onset ratio, S3-A symmetric")
    non = [c["values"]["medNon"] for c in sa_sym]
    bk.add("rfSthreeNonLo", g(min(non)), min(non), fs(sa_sym), "min median N_on, S3-A symmetric")
    bk.add("rfSthreeNonHi", g(max(non)), max(non), fs(sa_sym), "max median N_on, S3-A symmetric")
    onf = [c["values"]["onFinite"] for c in sa_sym]
    bk.add("rfSthreeOnFinLo", f(min(onf), 2), min(onf), fs(sa_sym), "min share N_on finite, S3-A symmetric")
    pe = max(c["values"]["predErr"] for c in sa_one + SB)
    bk.add("rfSthreePredErrMax", f(pe, 3), pe, fs(sa_one + SB), "max predErr over R3 cells")
    for te, nm in (("in+", "InPlus"), ("in-", "InMinus")):
        cs = [c for c in SB if c["params"]["te"] == te]
        ob = [c["values"]["medAucEnd"] for c in cs]
        pr = [c["values"]["medPred"] for c in cs]
        bk.add(f"rf{nm}ObsLo", f(min(ob), 2), min(ob), fs(cs), f"min medAucEnd, S3-B {te}")
        bk.add(f"rf{nm}ObsHi", f(max(ob), 2), max(ob), fs(cs), f"max medAucEnd, S3-B {te}")
        bk.add(f"rf{nm}PredLo", f(min(pr), 2), min(pr), fs(cs), f"min medPred, S3-B {te}")
        bk.add(f"rf{nm}PredHi", f(max(pr), 2), max(pr), fs(cs), f"max medPred, S3-B {te}")
    for te, nm in (("in+", "InPlus"), ("in-", "InMinus"), ("rev10", "RevTen"), ("rev40", "RevForty")):
        for d in ("ee", "oe"):
            c = next(c for c in SB if c["params"]["te"] == te and c["params"]["dict"] == d)
            v = c["values"]["pos"]
            bk.add(f"rf{nm}Pos{d.capitalize()}", f(v, 2), v, [c["file"]], f"share K_t > 0, S3-B {te} {d}")
    rev_ee = [c for c in SB if c["params"]["te"].startswith("rev") and c["params"]["dict"] == "ee"]
    re_ = min(c["values"]["medAucEnd"] for c in rev_ee)
    bk.add("rfRevEeEndMin", f(re_, 2), re_, fs(rev_ee), "min medAucEnd, ee reverse splits")
    allx = SA + SB
    wm = min(c["values"]["medAucWEnd"] for c in allx)
    bk.add("rfSthreeWinsMin", f(wm, 3), wm, fs(allx), "min winsorized medAucEnd, all S3 crossover cells")
    cm = min(c["values"]["medCostW10"] for c in allx)
    bk.add("rfSthreeCostMin", f(cm, 3), cm, fs(allx), "min medCostW10, all S3 crossover cells")
    for d, nm in (("oo4", "Same"), ("oe4", "Opp")):
        v = SC[d]["values"]
        bk.add(f"rfRateSlope{nm}", f(v["slope"], 2), v["slope"], [SC[d]["file"]], f"S3-C slope, {d}")
    bk.add("rfRateKs", f(SC["oo4"]["values"]["ks_H"], 3), SC["oo4"]["values"]["ks_H"], [SC["oo4"]["file"]],
           "S3-C ks_H, oo4, n = 512000")
    bk.add("rfRateLock", f(SC["oo4"]["values"]["lock_n512000"], 2), SC["oo4"]["values"]["lock_n512000"],
           [SC["oo4"]["file"]], "S3-C lock, oo4, n = 512000")
    bk.add("rfRatePos", f(SC["oe4"]["values"]["pos_n512000"], 2), SC["oe4"]["values"]["pos_n512000"],
           [SC["oe4"]["file"]], "S3-C P(K_t>0), oe4, n = 512000")

    # ---------------- run 4 (S4, DESIGN_S4.md)
    s4_macros(bk, U, fs)

    # ---------------- worst control values (Appendix B), runs 1 and 2
    C1c = {c["params"]["ctl"]: c for c in c1 if c["params"].get("block") == "control"}
    e1 = abs(C1c["C1"]["values"]["err"])
    bk.add("rfCtlCone", sci(e1), e1, [C1c["C1"]["file"]], "C1 |err|, CF engine vs normal formula")
    e3 = C1c["C3"]["values"]["maxerr"]
    bk.add("rfCtlCthree", sci(e3), e3, [C1c["C3"]["file"]], "C3 maxerr, stable CDF vs scipy")
    e10 = C2["C10"]["values"]["maxerr"]
    bk.add("rfCtlCten", f(e10, 4), e10, [C2["C10"]["file"]], "C10 maxerr, LePage vs stable CDF")
    zs = [(abs(C1c["C2"]["values"][k]), C1c["C2"]["file"]) for k in ("z50", "z2000")]
    zs += [(abs(C1c["C5"]["values"]["maxz"]), C1c["C5"]["file"])]
    zs += [(abs(C2["C6"]["values"][k]), C2["C6"]["file"]) for k in ("z100", "z1000")]
    zs += [(abs(C2["C7"]["values"][k]), C2["C7"]["file"]) for k in ("maxz_tail", "maxz_mom")]
    zm = max(z for z, _ in zs)
    bk.add("rfCtlZmax", f(zm, 2), zm, [fl for _, fl in zs], "max |z| over C2, C5, C6, C7")

    # ---------------- design constants with exponents (prose may not carry 10^{..} literals)
    bk.macros += [r"\newcommand{\rfNgridMaxB}{10^{18}}",
                  r"\newcommand{\rfNmaxS}{10^{4.5}}",
                  r"\newcommand{\rfNsymNmax}{2\cdot 10^{4}}",
                  # registered control tolerances (spec_h1v4.py C1, C3; spec_s2.py C7, C10)
                  r"\newcommand{\rfTolCone}{10^{-6}}",
                  r"\newcommand{\rfTolCthree}{10^{-3}}",
                  r"\newcommand{\rfTolCten}{0.005}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    head = "% generated by study/reframe/make_tex_reframe.py -- do not edit\n"
    (out_dir / "numbers.tex").write_text(head + "\n".join(bk.macros) + "\n")
    (out_dir / "tab_criteria.tex").write_text(head + tab_criteria(A, B, X, T, p1_pass, p4_pass, q1, SA, SB, SC, U))
    (out_dir / "tab_fast.tex").write_text(head + tab_fast(A, B))
    (out_dir / "tab_slow.tex").write_text(head + tab_slow(A, B))
    (out_dir / "tab_s2.tex").write_text(head + tab_s2(X))
    (out_dir / "tab_s3a.tex").write_text(head + tab_s3a(SA))
    (out_dir / "tab_s3b.tex").write_text(head + tab_s3b(SA, SB))
    (out_dir / "tab_s4a.tex").write_text(head + tab_s4a(U))
    (out_dir / "tab_s4b.tex").write_text(head + tab_s4b(U))
    (out_dir / "tab_s4c.tex").write_text(head + tab_s4c(U))
    if claims_path is not None:
        write_claims(claims_path, bk)


def sci(x: float) -> str:
    e = int(f"{x:.1e}".split("e")[1])
    m = x / 10 ** e
    return rf"{m:.1f}\cdot 10^{e}" if 0 <= e < 10 else rf"{m:.1f}\cdot 10^{{{e}}}"


# ----------------------------------------------------------------------------- run 4 (S4)

def s4_d1(p: dict) -> bool:
    """Cells of Q2 (run 2) and R2 (run 3), as registered in DESIGN_S4.md."""
    if p["run"] == 2:
        return p["te"] == "sym" and p["src"] == "gauss"
    return p["oblock"] == "A" and p["te"] == "sym"


def s4_d2(p: dict) -> bool:
    """Cells of Q3 (run 2) and R3 (run 3)."""
    if p["run"] == 2:
        return p["te"] == "one"
    return p["oblock"] == "B" or p["te"] == "one"


def s4_r12(v: dict) -> bool:
    return (v["medPeak"] >= 0.80 and v["onFinite"] >= 0.80 and v["medNon"] <= 2e4
            and 0.67 <= v["ratio"] <= 1.5)


def s4_verdicts(U) -> list[tuple[str, str, int, int, bool]]:
    """(name, text, cells passing, cells, verdict) for the registered criteria of DESIGN_S4.md."""
    D1 = [c for c in U["D"] if s4_d1(c["params"])]
    D2 = [c for c in U["D"] if s4_d2(c["params"])]
    a_th = [c for c in U["A4"] if c["params"]["dict"] != "poly3"]
    a_sym = [c for c in a_th if c["params"]["te"].startswith("sym")]
    a_one = [c for c in a_th if c["params"]["te"].startswith("one")]
    c_sym = [c for c in U["C4"] if c["params"]["te"] == "sym"]
    c_one = [c for c in U["C4"] if c["params"]["te"] == "one"]
    e_th = [c for c in U["E4"] if c["params"]["te"] != "rev40"]
    v = lambda c: c["values"]
    side = lambda c: v(c)["medAucEnd"] < 0.5 if v(c)["beta_te"] > 0 else v(c)["medAucEnd"] > 0.5
    rows = [
        ("D0", r"runs 2--3 re-run with their seeds: the stored medians are reproduced",
         sum(v(c)["reproMaxDiff"] <= 1e-9 for c in U["D"]), len(U["D"]), "all"),
        ("D1", r"Q2/R2 cells: per-fit onset ratio, $q_{25}\ge 0.5$ and $q_{75}\le 2$",
         sum(v(c)["ratioQ"][0] >= 0.5 and v(c)["ratioQ"][2] <= 2.0 for c in D1), len(D1), "80"),
        ("D2", r"Q3/R3 cells: per-fit $|\mathrm{AUC}_{N_{\max}}-\mathrm{pred}|$, $p_{90}\le 0.15$",
         sum(v(c)["errQ"][3] <= 0.15 for c in D2), len(D2), "80"),
        ("D3", r"Q2/R2 cells: fixed $N\ge 10^3$, $|\text{mean AUC}_N-\text{mean }\Psi|\le 0.05$",
         sum(v(c)["fwMaxDiff"] <= 0.05 for c in D1), len(D1), "80"),
        ("A1", r"$(x,x^3)$, $(x,|x|^3)$, $\alpha_c\in\{2.2,2.5\}$, symmetric: as R1 and R2",
         sum(s4_r12(v(c)) for c in a_sym), len(a_sym), "all"),
        ("A2", r"$(x,|x|^3)$, one-sided: as R3", sum(v(c)["predErr"] <= 0.10 for c in a_one), len(a_one), "all"),
        ("B1", r"$\dict{ee}$, $n=1000$, $\bte=\pm\tfrac12$: lock $\ge 0.95$, as R3, side of $\tfrac12$",
         sum(v(c)["lock"] >= 0.95 and v(c)["predErr"] <= 0.10 and side(c) for c in U["B4"]), len(U["B4"]), "all"),
        ("C1", r"RflyMAD, $\alpha_c\in\{1.2,1.5\}$, symmetric: as R1 and R2",
         sum(s4_r12(v(c)) for c in c_sym), len(c_sym), "all"),
        ("C2", r"RflyMAD, one-sided: as R3", sum(v(c)["predErr"] <= 0.10 for c in c_one), len(c_one), "all"),
        ("C3", r"RflyMAD, symmetric: as R4",
         sum(v(c)["medAucWEnd"] >= 0.90 and v(c)["medCostW10"] >= -0.02 for c in c_sym), len(c_sym), "all"),
        ("E1", r"scale-only clean laws, $\mu^{\mathrm{te}}_L>0$ splits: as R3",
         sum(v(c)["predErr"] <= 0.10 for c in e_th), len(e_th), "all"),
    ]
    return [(k, t, ok, n, ok == n if rule == "all" else ok >= 0.8 * n) for k, t, ok, n, rule in rows]


def s4_macros(bk: Book, U, fs) -> None:
    D1 = [c for c in U["D"] if s4_d1(c["params"])]
    D2 = [c for c in U["D"] if s4_d2(c["params"])]
    for k, _, ok, n, _ in s4_verdicts(U):
        nm = {"D0": "Dzero", "D1": "Done", "D2": "Dtwo", "D3": "Dthree", "A1": "Aone", "A2": "Atwo",
              "B1": "Bone", "C1": "Cone", "C2": "Ctwo", "C3": "Cthree", "E1": "Eone"}[k]
        src = U["D"] if k == "D0" else D1 if k in ("D1", "D3") else D2 if k == "D2" else \
            U[{"A": "A4", "B": "B4", "C": "C4", "E": "E4"}[k[0]]]
        bk.add(f"rfS{nm}Pass", str(ok), ok, fs(src), f"count {k} pass, run 4")
        bk.add(f"rfS{nm}N", str(n), n, fs(src), f"count {k} cells, run 4")
    vals = lambda cs, fn: [fn(c["values"]) for c in cs]
    for nm, cs, fn, d, how in (
            ("DoneQlo", D1, lambda v: v["ratioQ"][0], 2, "min per-fit q25 of the onset ratio, D1 cells"),
            ("DoneQloMax", D1, lambda v: v["ratioQ"][0], 2, "max per-fit q25 of the onset ratio, D1 cells"),
            ("DoneQhi", D1, lambda v: v["ratioQ"][2], 2, "max per-fit q75 of the onset ratio, D1 cells"),
            ("DoneQhiMin", D1, lambda v: v["ratioQ"][2], 2, "min per-fit q75 of the onset ratio, D1 cells"),
            ("DtwoPninety", D2, lambda v: v["errQ"][3], 3, "max per-fit p90 of |AUC_end - pred|, D2 cells"),
            ("DtwoMax", D2, lambda v: v["errMax"], 2, "max per-fit |AUC_end - pred|, D2 cells"),
            ("DthreeMax", D1, lambda v: v["fwMaxDiff"], 3, "max fwMaxDiff, D1 cells")):
        xs = vals(cs, fn)
        x = min(xs) if nm in ("DoneQlo", "DoneQhiMin") else max(xs)
        bk.add(f"rfS{nm}", f(x, d), x, fs(cs), how)
    dmed = sorted(vals(D1, lambda v: v["fwMaxDiff"]))[len(D1) // 2]
    bk.add("rfSDthreeMed", f(dmed, 3), dmed, fs(D1), "median fwMaxDiff, D1 cells")
    a_th = [c for c in U["A4"] if c["params"]["dict"] != "poly3"]
    a_sym = [c for c in a_th if c["params"]["te"].startswith("sym")]
    a_one = [c for c in a_th if c["params"]["te"].startswith("one")]
    for nm, cs, key, d, agg in (("ARatioLo", a_sym, "ratio", 2, min), ("ARatioHi", a_sym, "ratio", 2, max),
                                ("ANonLo", a_sym, "medNon", None, min), ("ANonHi", a_sym, "medNon", None, max),
                                ("APredErrMax", a_one, "predErr", 3, max),
                                ("APosLo", [c for c in a_th if c["params"]["dict"] == "oe3"], "pos", 2, min),
                                ("APosHi", [c for c in a_th if c["params"]["dict"] == "oe3"], "pos", 2, max),
                                ("ALockOo", [c for c in a_th if c["params"]["dict"] == "oo3"], "lock", 2, max)):
        x = agg(c["values"][key] for c in cs)
        bk.add(f"rfS{nm}", g(x) if d is None else f(x, d), x, fs(cs), f"{agg.__name__} {key}, run-4 block A")
    import math as _m
    dec = [_m.log10(c["values"]["NstarQ"][2] / c["values"]["NstarQ"][0]) for c in a_sym]
    bk.add("rfSANstarDecLo", f(min(dec), 1), min(dec), fs(a_sym), "min log10(q75/q25) of per-fit N*, A symmetric")
    bk.add("rfSANstarDecHi", f(max(dec), 1), max(dec), fs(a_sym), "max log10(q75/q25) of per-fit N*, A symmetric")
    for nm, key, d in (("OnFin", "onFinite", 2), ("Peak", "medPeak", 2), ("ErrSym", "predErr", 3),
                       ("Fw", "fwMaxDiff", 3)):
        xs = [c["values"][key] for c in a_sym]
        bk.add(f"rfSA{nm}Lo", f(min(xs), d), min(xs), fs(a_sym), f"min {key}, A symmetric")
        bk.add(f"rfSA{nm}Hi", f(max(xs), d), max(xs), fs(a_sym), f"max {key}, A symmetric")
    for c in a_sym:
        p, v = c["params"], c["values"]
        if p["dict"] == "oo3":
            al = {2.2: "TwoTwo", 2.5: "TwoFive"}[p["alpha"]]
            for i, kn in ((1, "Med"), (0, "Qlo"), (2, "Qhi")):
                bk.add(f"rfSAOo{al}Ratio{kn}", f(v["ratioQ"][i], 2), v["ratioQ"][i], [c["file"]],
                       f"per-fit onset ratio quantile {i}, oo3 alpha_c = {p['alpha']}")
    fdg = RES / "diag_s4_clean_drift.json"
    dg = json.loads(fdg.read_text())
    fdgs = [f"study/reframe/results/{fdg.name}"]
    eff = [r["ratio_eff_q"][1] for r in dg.values()]
    bk.add("rfSAEffLo", f(min(eff), 2), min(eff), fdgs, "min median onset ratio with N*_eff, A symmetric (post-hoc)")
    bk.add("rfSAEffHi", f(max(eff), 2), max(eff), fdgs, "max median onset ratio with N*_eff, A symmetric (post-hoc)")
    oe_eff = [r["ratio_eff_q"][1] for k, r in dg.items() if k.startswith("oe3")]
    bk.add("rfSAOeEffLo", f(min(oe_eff), 2), min(oe_eff), fdgs, "min median N*_eff onset ratio, oe3 (post-hoc)")
    bk.add("rfSAOeEffHi", f(max(oe_eff), 2), max(oe_eff), fdgs, "max median N*_eff onset ratio, oe3 (post-hoc)")
    oe_r = [c["values"]["ratio"] for c in a_sym if c["params"]["dict"] == "oe3"]
    bk.add("rfSAOeRatioLo", f(min(oe_r), 1), min(oe_r), fs(a_sym), "min median onset ratio, oe3 symmetric")
    bk.add("rfSAOeRatioHi", f(max(oe_r), 0), max(oe_r), fs(a_sym), "max median onset ratio, oe3 symmetric")
    for c in U["E4"]:
        nm = {"fwd10": "FwdTen", "rev10": "RevTen", "fwd40": "FwdForty", "rev40": "RevForty"}[c["params"]["te"]]
        v = c["values"]
        ks = c["values"]["ksub_lstar"]
        lk = sorted(e for k, e in zip(v["fit_Kt"], v["fit_end"]) if k * ks < 0)
        ul = sorted(e for k, e in zip(v["fit_Kt"], v["fit_end"]) if k * ks > 0)
        med = lambda xs: (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2
        bk.add(f"rfSE{nm}LockedEnd", f(med(lk), 2), med(lk), [c["file"]], f"median AUC_end, locked fits, E {c['params']['te']}")
        bk.add(f"rfSE{nm}UnlockedEnd", f(med(ul), 2), med(ul), [c["file"]], f"median AUC_end, unlocked fits, E {c['params']['te']}")
        bk.add(f"rfSE{nm}LockedN", str(len(lk)), len(lk), [c["file"]], f"locked fits, E {c['params']['te']}")
        bk.add(f"rfSE{nm}SubEnd", f(v["medAucSEnd"], 2), v["medAucSEnd"], [c["file"]], f"medAucSEnd, E {c['params']['te']}")
    for c in [c for c in U["A4"] if c["params"]["dict"] == "poly3"]:
        al = {2.2: "TwoTwo", 2.5: "TwoFive"}[c["params"]["alpha"]]
        v = c["values"]
        for key, nm, d in (("medPeak", "Peak", 2), ("onFinite", "OnFin", 2), ("medNon", "Non", None),
                           ("medAucEnd", "End", 2), ("medAucWEnd", "EndW", 3), ("pos_K3", "PosK", 2)):
            bk.add(f"rfSPoly{al}{nm}", g(v[key]) if d is None else f(v[key], d), v[key], [c["file"]],
                   f"{key}, poly3, alpha_c = {c['params']['alpha']}")
    for c in U["B4"]:
        nm = {"in+": "Plus", "in-": "Minus"}[c["params"]["te"]]
        v = c["values"]
        for key, kn, d in (("lock", "Lock", 2), ("medAucEnd", "End", 2), ("medPred", "Pred", 2),
                           ("predErr", "Err", 3)):
            bk.add(f"rfSB{nm}{kn}", f(v[key], d), v[key], [c["file"]], f"{key}, run-4 block B {c['params']['te']}")
    c_sym = [c for c in U["C4"] if c["params"]["te"] == "sym"]
    c_one = [c for c in U["C4"] if c["params"]["te"] == "one"]
    for nm, cs, key, d, agg in (("CRatioLo", c_sym, "ratio", 2, min), ("CRatioHi", c_sym, "ratio", 2, max),
                                ("CNonLo", c_sym, "medNon", None, min), ("CNonHi", c_sym, "medNon", None, max),
                                ("COnFinLo", c_sym, "onFinite", 2, min),
                                ("CPredErrMax", c_one, "predErr", 3, max),
                                ("CEndOneMax", [c for c in c_one if c["params"]["dict"] == "ee"], "medAucEnd", 2, max),
                                ("CWinsMin", U["C4"], "medAucWEnd", 3, min),
                                ("CCostMin", c_sym, "medCostW10", 3, min)):
        x = agg(c["values"][key] for c in cs)
        bk.add(f"rfS{nm}", g(x) if d is None else f(x, d), x, fs(cs), f"{agg.__name__} {key}, run-4 block C")
    for c in U["E4"]:
        nm = {"fwd10": "FwdTen", "rev10": "RevTen", "fwd40": "FwdForty", "rev40": "RevForty"}[c["params"]["te"]]
        v = c["values"]
        for key, kn, d in (("medAucEnd", "End", 2), ("medPred", "Pred", 2), ("predErr", "Err", 3),
                           ("pos", "Pos", 2), ("medAucWEnd", "EndW", 3), ("mu_te_L", "Mu", 4)):
            bk.add(f"rfSE{nm}{kn}", f(v[key], d), v[key], [c["file"]], f"{key}, run-4 block E {c['params']['te']}")


DTEX4 = {"oo3": r"$(x, x^3)$", "oe3": r"$(x, |x|^3)$", "poly3": r"$(x, x^2, x^3)$",
         "ee": r"$\mathsf{ee}$", "oe": r"$\mathsf{oe}$"}
TE4 = {"sym05": r"$(0.05, 0.05)$", "sym20": r"$(0.2, 0.2)$", "one10": r"$(0.1, 0)$", "one40": r"$(0.4, 0)$",
       "in+": r"$(0.075, 0.025)$", "in-": r"$(0.025, 0.075)$", "sym": r"$(0.05, 0.05)$", "one": r"$(0.1, 0)$",
       "fwd10": r"$(0.1, 0)$", "rev10": r"$(0, 0.1)$", "fwd40": r"$(0.4, 0)$", "rev40": r"$(0, 0.4)$"}


def _rng(xs, d) -> str:
    lo, hi = min(xs), max(xs)
    return f(lo, d) if f(lo, d) == f(hi, d) else rf"{f(lo, d)}\,\ldots\,{f(hi, d)}"


def tab_s4a(U) -> str:
    """Run 4, block D: per-fit dispersion and the fixed-window check, by group of re-run cells."""
    groups = (
        ("Q2: run 2, sym., Gaussian",
         lambda p: p["run"] == 2 and p["te"] == "sym" and p["src"] == "gauss"),
        ("run 2, sym., RflyMAD", lambda p: p["run"] == 2 and p["te"] == "sym" and p["src"] == "rfly"),
        ("R2: run 3, sym.", lambda p: p["run"] == 3 and p["oblock"] == "A" and p["te"] == "sym"),
        ("Q3: run 2, one-sided", lambda p: p["run"] == 2 and p["te"] == "one"),
        ("R3: run 3, skewed",
         lambda p: p["run"] == 3 and (p["oblock"] == "B" or p["te"] == "one")),
    )
    body = [r"\begin{tabular}{@{}lrccccc@{}}", r"\toprule",
            r"cells & & \multicolumn{3}{c}{onset ratio over fits} & $p_{90}$ of & fixed \\",
            r"\cmidrule(lr){3-5}",
            r"& number & median & $q_{25}$ & $q_{75}$ & per-fit error & window \\",
            r"\midrule"]
    for lab, sel in groups:
        cs = [c["values"] for c in U["D"] if sel(c["params"])]
        fin = [v for v in cs if v["ratioQ"][0] == v["ratioQ"][0]]
        sym_rows = "sym." in lab  # the ratio is the registered yardstick only under symmetric splits
        r = lambda i: _rng([v["ratioQ"][i] for v in fin], 2) if fin and sym_rows else "--"
        body.append(rf"{lab} & {len(cs)} & {r(1)} & {r(0)} & {r(2)} & {_rng([v['errQ'][3] for v in cs], 3)} & "
                    rf"{_rng([v['fwMaxDiff'] for v in cs], 3)} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_s4b(U) -> str:
    """Run 4, blocks A, B, E: one row per cell."""
    body = [r"\begin{tabular}{@{}llrrlrcccccc@{}}", r"\toprule",
            r"block & dictionary & $\alpha_c$ & $n$ & test split & $\bte$ & $\hat P\{\hat K_t>0\}$ & peak & ratio & "
            r"$\mathrm{AUC}_{N_{\max}}$ & pred. & error, median / $p_{90}$ \\", r"\midrule"]
    last = None
    for b, lab in (("A4", "A"), ("B4", "B"), ("E4", "E")):
        for c in U[b]:
            p, v = c["params"], c["values"]
            if last is not None and b != last:
                body.append(r"\addlinespace")
            last = b
            d = p.get("dict", "ee")
            al = p.get("alpha", 1.2 if b == "B4" else 3.0)
            n = p.get("n", 1000 if b == "B4" else 200)
            if d == "poly3":
                body.append(rf"{lab} & {DTEX4[d]} & {al:g} & {n} & {TE4[p['te']]} & -- & {f(v['pos_K3'], 2)} & "
                            rf"{f(v['medPeak'], 2)} & -- & {f(v['medAucEnd'], 2)} & -- & -- \\")
                continue
            rat = f(v["ratio"], 2) if v["ratio"] == v["ratio"] and not p["te"].startswith(("one", "in", "fwd", "rev")) else "--"
            body.append(rf"{lab} & {DTEX4[d]} & {al:g} & {n} & {TE4[p['te']]} & {v['beta_te']:+.1f} & {f(v['pos'], 2)} & "
                        rf"{f(v['medPeak'], 2)} & {rat} & {f(v['medAucEnd'], 2)} & "
                        + (r"-- & -- \\" if v["mu_te_L"] <= 0 else  # outside Theorem 7 (mu^te_L <= 0)
                           rf"{f(v['medPred'], 2)} & {f(v['predErr'], 3)} / {f(v['errQ'][3], 3)} \\"))
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_s4c(U) -> str:
    """Run 4, block C: RflyMAD clean pools in the run-3 regime."""
    body = [r"\begin{tabular}{@{}rrlrlcccrcccc@{}}", r"\toprule",
            r"$\alpha_c$ & $a^*$ & dict. & $n$ & split & $\hat P\{\hat K_t>0\}$ & peak & onset found & median $N_{\mathrm{on}}$ & ratio & "
            r"$\mathrm{AUC}_{N_{\max}}$ & error, med. / $p_{90}$ & winsorized \\", r"\midrule"]
    rows = sorted(U["C4"], key=lambda c: (c["params"]["alpha"], c["params"]["astar"], c["params"]["n"],
                                          c["params"]["te"], c["params"]["dict"]))
    last = None
    for c in rows:
        p, v = c["params"], c["values"]
        grp = (p["alpha"], p["astar"], p["n"])
        if last is not None and grp != last:
            body.append(r"\addlinespace")
        last = grp
        sym = p["te"] == "sym"
        non = g(v["medNon"]) if v["medNon"] == v["medNon"] else "--"
        err = (f(v["predErr"], 3) + " / " + f(v["errQ"][3], 3)) if not sym else "--"
        body.append(rf"{p['alpha']:g} & {p['astar']:g} & {DTEX4[p['dict']]} & {p['n']} & {p['te']} & {f(v['pos'], 2)} & "
                    rf"{f(v['medPeak'], 2)} & {f(v['onFinite'], 2)} & {non} & {f(v['ratio'], 2) if sym else '--'} & "
                    rf"{f(v['medAucEnd'], 2)} & {err} & {f(v['medAucWEnd'], 3)} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_criteria(A, B, X, T, p1, p4, q1, SA, SB, SC, U) -> str:
    same = [c for c in A if c["values"]["same_parity"] == 1.0]
    p2 = sum(c["values"]["dauc_n128000"] <= 0.01 and c["values"]["dauc_n128000"] < c["values"]["dauc_n500"] for c in A)
    p3 = sum(c["values"]["lock_n128000"] >= 0.95 for c in same)
    p6 = sum(abs(B[d]["values"]["slope_err"]) <= 0.15 for d in B)
    sym = [c for c in X if c["params"]["te"] == "sym"]
    one = [c for c in X if c["params"]["te"] == "one"]
    q2c = [c for c in sym if c["params"]["src"] != "rfly"]
    q2 = sum(0.67 <= c["values"]["ratio"] <= 1.5 for c in q2c)
    q3 = sum(c["values"]["predErr"] <= 0.10 for c in one)
    q4 = int(T["ee"]["values"]["lock_n128000"] >= 0.95) + sum(T[d]["values"]["dauc_n128000"] <= 0.01 for d in T)
    q5a = sum(c["values"]["medAucWEnd"] >= 0.90 for c in sym)
    q5b = sum(c["values"]["medCostW10"] >= -0.02 for c in sym)
    rows = [
        ("P1", r"slope of median $|\hat K_t|$ in $n$ within $0.10$ of $-(a^*-a_L)/\alpha_c$", p1, len(A)),
        ("P2", r"median $|\Delta\mathrm{AUC}_{10}|\le 0.01$ at the largest $n$", p2, len(A)),
        ("P3", r"same-parity lock frequency $\ge 0.95$ at the largest $n$", p3, len(same)),
        ("P4", r"opposite parity, $p_+=\tfrac12$: $|\hat P\{\hat K_t>0\}-\tfrac12|\le 0.087$", p4, 12),
        ("P6", r"slope of median $N_{\mathrm{on}}$ in $n$ within $0.15$ of $(a^*-a_L)/(a^*-\alpha_c)$", p6, len(B)),
        ("Q1", r"peak $\ge 0.80$, $N_{\mathrm{on}}$ finite in $\ge 80\%$, median $N_{\mathrm{on}}\le 2\cdot 10^4$", q1, len(sym)),
        ("Q2", r"$N_{\mathrm{on}}/(\lambda_m^{\alpha_c/(a^*-\alpha_c)}N^*_n(\hat K))\in[0.67,1.5]$", q2, len(q2c)),
        ("Q3", r"one-sided: $|\mathrm{AUC}_{N_{\max}}-\mathrm{pred}|\le 0.10$", q3, len(one)),
        ("Q4", r"Student-$t_3$ contaminant: lock $\ge 0.95$, $|\Delta\mathrm{AUC}_{10}|\le 0.01$", q4, 3),
        ("Q5a", r"winsorized $\mathrm{AUC}_{N_{\max}}\ge 0.90$", q5a, len(sym)),
        ("Q5b", r"winsorized minus sub-dictionary $\mathrm{AUC}_{10}\ge -0.02$ (clean)", q5b, len(sym)),
    ]
    s_sym = [c for c in SA if c["params"]["te"] == "sym"]
    s_r3 = [c for c in SA if c["params"]["te"] == "one"] + SB
    sv = lambda c: c["values"]
    rows += [
        ("R1", r"as Q1, $\alpha_c\in\{1.2,1.5\}$, $a^*\in\{3,4\}$",
         sum(sv(c)["medPeak"] >= 0.80 and sv(c)["onFinite"] >= 0.80 and sv(c)["medNon"] <= 2e4 for c in s_sym), len(s_sym)),
        ("R2", r"as Q2, same cells", sum(0.67 <= sv(c)["ratio"] <= 1.5 for c in s_sym), len(s_sym)),
        ("R3", r"as Q3, one-sided, interior $\bte$ and reverse splits", sum(sv(c)["predErr"] <= 0.10 for c in s_r3), len(s_r3)),
        ("R4", r"as Q5a and Q5b", sum(sv(c)["medAucWEnd"] >= 0.90 and sv(c)["medCostW10"] >= -0.02 for c in s_sym), len(s_sym)),
        ("R5", r"two columns, equal scales: slope within $0.10$ of $-(a^*-a_L)/\alpha_c$",
         sum(abs(sv(c)["slope_err"]) <= 0.10 for c in SC.values()), len(SC)),
        ("R6", r"same parity: lock $\ge 0.95$, Kolmogorov distance to $H\le 0.08$",
         int(sv(SC["oo4"])["lock_n512000"] >= 0.95 and sv(SC["oo4"])["ks_H"] <= 0.08), 1),
        ("R7", r"opposite parity: $|\hat P\{\hat K_t>0\}-\tfrac12|\le 0.087$",
         int(abs(sv(SC["oe4"])["pos_n512000"] - 0.5) <= 0.087), 1),
    ]
    body = [r"\begin{tabular}{@{}llrl@{}}", r"\toprule",
            r"& criterion (registered before the run) & cells & verdict \\", r"\midrule"]
    for i, (k, txt, ok, tot) in enumerate(rows):
        if k in ("Q1", "R1"):
            body.append(r"\midrule")
        body.append(rf"{k} & {txt} & {ok}/{tot} & {'pass' if ok == tot else r'\textbf{fail}'} \\")
    body.append(r"\midrule")
    for k, txt, ok, tot, verdict in s4_verdicts(U):
        body.append(rf"{k} & {txt} & {ok}/{tot} & {'pass' if verdict else r'\textbf{fail}'} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_fast(A, B) -> str:
    body = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"dictionary & parity & max $|\Delta\mathrm{AUC}_{10}|$ & min lock & $N_{\mathrm{on}}/(N^*\lambda_m^3)$ & max $|\mathrm{AUC}-\rho|$ \\",
            r"& $(l^*,t)$ & $n=128\,000$ & $n=128\,000$ & $n=1000$ & past crossover \\", r"\midrule"]
    for d in DICT_ORDER:
        cs = [c for c in A if c["params"]["dict"] == d]
        same = cs[0]["values"]["same_parity"] == 1.0
        dm = max(c["values"]["dauc_n128000"] for c in cs)
        lock = f(min(c["values"]["lock_n128000"] for c in cs), 2) if same else "--"
        v = B[d]["values"]
        s2 = max(v[f"S2err_n{n}"] for n in (50, 200, 1000))
        body.append(rf"{DICT_TEX[d]} & {'same' if same else 'opposite'} & {f(dm, 4)} & {lock} & {f(v['ratio_n1000'], 3)} & {f(s2, 4)} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_slow(A, B) -> str:
    body = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
            r"& \multicolumn{2}{c}{slope of median $|\hat K_t|$ in $n$} & \multicolumn{2}{c}{$\hat P\{\hat K_t>0\}$, $p_+=\tfrac12$, $n=128\,000$} & \multicolumn{2}{c}{onset slope in $n$} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
            r"dictionary & $a^*=12/5$ (theory) & $a^*=4$ (theory) & $a^*=12/5$ & $a^*=4$ & measured & theory \\", r"\midrule"]
    for d in DICT_ORDER:
        cells = []
        for a_ in (2.4, 4.0):
            cs = [c for c in A if c["params"]["dict"] == d and c["params"]["astar"] == a_]
            sl = [c["values"]["slope"] for c in cs]
            cells.append(rf"{f(min(sl), 2)}\,\ldots\,{f(max(sl), 2)} ({f(cs[0]['values']['slope_pred'], 2)})")
        pos = []
        for a_ in (2.4, 4.0):
            cs = [c for c in A if c["params"]["dict"] == d and c["params"]["astar"] == a_ and c["params"]["pplus"] == 0.5]
            ps = [c["values"]["pos_n128000"] for c in cs]
            pos.append(rf"{f(min(ps), 2)}\,\ldots\,{f(max(ps), 2)}")
        v = B[d]["values"]
        body.append(rf"{DICT_TEX[d]} & {cells[0]} & {cells[1]} & {pos[0]} & {pos[1]} & {f(v['slope'], 2)} & {f(v['slope_pred'], 2)} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_s3a(SA) -> str:
    """Run 3, block A, symmetric test contamination: one row per cell."""
    body = [r"\begin{tabular}{@{}rrlrccrrcc@{}}", r"\toprule",
            r"$\alpha_c$ & $a^*$ & dict. & $n$ & peak & onset found & median $N_{\mathrm{on}}$ & median $N^*_n(\hat K)$ & ratio & winsorized \\",
            r"\midrule"]
    rows = sorted([c for c in SA if c["params"]["te"] == "sym"],
                  key=lambda c: (c["params"]["alpha"], c["params"]["astar"], c["params"]["n"],
                                 DICT_ORDER.index(c["params"]["dict"])))
    last = None
    for c in rows:
        p, v = c["params"], c["values"]
        grp = (p["alpha"], p["astar"], p["n"])
        if last is not None and grp != last:
            body.append(r"\addlinespace")
        last = grp
        body.append(rf"{p['alpha']:g} & {p['astar']:g} & {DICT_TEX[p['dict']]} & {p['n']} & {v['medPeak']:.2f} & "
                    rf"{v['onFinite']:.2f} & {g(v['medNon'])} & {g(v['medNstar'])} & {v['ratio']:.2f} & {v['medAucWEnd']:.3f} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_s3b(SA, SB) -> str:
    """Run 3, cells of criterion R3: skewed test contamination. Per-fit limit check."""
    te_tex = {"one": r"$(0.10, 0)$", "in+": r"$(0.3, 0.1)$", "in-": r"$(0.1, 0.3)$",
              "rev10": r"$(0, 0.1)$", "rev40": r"$(0, 0.4)$"}
    body = [r"\begin{tabular}{@{}rrlrlrcccc@{}}", r"\toprule",
            r"$\alpha_c$ & $a^*$ & dict. & $n$ & test split & $\bte$ & $\hat P\{\hat K_t > 0\}$ & median $\mathrm{AUC}_{N_{\max}}$ & median pred. & median error \\",
            r"\midrule"]
    rows = sorted([c for c in SA if c["params"]["te"] == "one"] + SB,
                  key=lambda c: (c["params"]["block"], c["params"]["alpha"], c["params"]["astar"], c["params"]["n"],
                                 list(te_tex).index(c["params"]["te"]), DICT_ORDER.index(c["params"]["dict"])))
    last = None
    for c in rows:
        p, v = c["params"], c["values"]
        grp = (p["block"], p["alpha"], p["astar"], p["n"])
        if last is not None and grp != last:
            body.append(r"\addlinespace")
        last = grp
        body.append(rf"{p['alpha']:g} & {p['astar']:g} & {DICT_TEX[p['dict']]} & {p['n']} & {te_tex[p['te']]} & "
                    rf"{v['beta_te']:+.1f} & {v['pos']:.2f} & {v['medAucEnd']:.2f} & {v['medPred']:.2f} & {v['predErr']:.3f} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def tab_s2(X) -> str:
    src_tex = {("gauss", "pareto"): "Gaussian, Pareto", ("gauss", "t3"): r"Gaussian, $t_3$", ("rfly", "pareto"): "RflyMAD, Pareto"}
    body = [r"\begin{tabular}{@{}llrccrrccc@{}}", r"\toprule",
            r"clean laws, contaminant & dict. & $n$ & peak & onset found & median $N_{\mathrm{on}}$ & median $N^*_n(\hat K)$ & ratio & $\mathrm{AUC}_{N_{\max}}$ & winsorized \\",
            r"\midrule"]
    key = lambda c: (("gauss", "pareto"), ("gauss", "t3"), ("rfly", "pareto")).index((c["params"]["src"], c["params"]["cont"]))
    sym = sorted([c for c in X if c["params"]["te"] == "sym"],
                 key=lambda c: (key(c), DICT_ORDER.index(c["params"]["dict"]), c["params"]["n"]))
    last = None
    for c in sym:
        p, v = c["params"], c["values"]
        grp = src_tex[(p["src"], p["cont"])]
        if last is not None and grp != last:
            body.append(r"\addlinespace")
        lab = grp if grp != last else ""
        last = grp
        body.append(rf"{lab} & {DICT_TEX[p['dict']]} & {p['n']} & {f(v['medPeak'], 3)} & {f(v['onFinite'], 2)} & {g(v['medNon'])} & {g(v['medNstar'])} & "
                    rf"{f(v['ratio'], 2)} & {f(v['medAucEnd'], 3)} & {f(v['medAucWEnd'], 3)} \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(body) + "\n"


def write_claims(path: Path, bk: Book) -> None:
    lines = ["version: 1", "manuscript: paper/reframe/main.tex", "results_root: study/reframe/results",
             "generator: study/reframe/make_tex_reframe.py",
             "note: every result number in the text is a macro of generated/numbers.tex or a cell of a generated table;",
             "  each entry below names the hashed artifacts and the aggregation that produced it.",
             "claims:"]
    for i, c in enumerate(bk.claims):
        lines += [f"  - id: m{i:03d}", f"    macro: \"\\\\{c['macro']}\"", f"    value: {c['value']!r}",
                  "    status: resolved", "    evidence:", "      kind: artifact",
                  f"      locator: \"{c['aggregation']}\"", "      artifacts:"]
        lines += [f"        - {p}" for p in c["files"]]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    build(OUT, PAPER / "claims.yaml")
    print("wrote", OUT, "and", PAPER / "claims.yaml")
