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
PAPER = HERE.parent.parent / "paper" / "reframe"
OUT = PAPER / "generated"
FILES = ("numbers.tex", "tab_criteria.tex", "tab_fast.tex", "tab_slow.tex", "tab_s2.tex")
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
    (out_dir / "tab_criteria.tex").write_text(head + tab_criteria(A, B, X, T, p1_pass, p4_pass, q1))
    (out_dir / "tab_fast.tex").write_text(head + tab_fast(A, B))
    (out_dir / "tab_slow.tex").write_text(head + tab_slow(A, B))
    (out_dir / "tab_s2.tex").write_text(head + tab_s2(X))
    if claims_path is not None:
        write_claims(claims_path, bk)


def sci(x: float) -> str:
    e = int(f"{x:.1e}".split("e")[1])
    m = x / 10 ** e
    return rf"{m:.1f}\cdot 10^{e}" if 0 <= e < 10 else rf"{m:.1f}\cdot 10^{{{e}}}"


def tab_criteria(A, B, X, T, p1, p4, q1) -> str:
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
    body = [r"\begin{tabular}{@{}llrl@{}}", r"\toprule",
            r"& criterion (registered before the run) & cells & verdict \\", r"\midrule"]
    for i, (k, txt, ok, tot) in enumerate(rows):
        if k == "Q1":
            body.append(r"\midrule")
        body.append(rf"{k} & {txt} & {ok}/{tot} & {'pass' if ok == tot else r'\textbf{fail}'} \\")
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
