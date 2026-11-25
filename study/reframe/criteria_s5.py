"""Evaluate the registered criteria F1-F3 and G1-G3 of DESIGN_S5.md on the hashed S5 artifacts, and
re-predict the one-sided A4 cells of run 4 with the compensated LePage sum (reported, not a
criterion).

Reads results/<run-id>/*.json and the stored run-4 cells; recomputes only the run-4 prediction.
Writes results/criteria_s5.json. Usage: python criteria_s5.py [run-id]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import h1v4 as H
import h1v4_s5 as G

HERE = Path(__file__).resolve().parent
RUN = sys.argv[1] if len(sys.argv) > 1 else "h1v4-s5-2026-11-24"
D = HERE / "results" / RUN
R4 = HERE / "results" / "h1v4-s4-2026-11-24"
OUT = HERE / "results" / "criteria_s5.json"


def load(d: Path, block: str) -> list[dict]:
    rows = []
    for f in sorted(d.glob("*.json")):
        if f.name == "RUN.json":
            continue
        a = json.loads(f.read_text())
        if a["params"].get("block") == block:
            rows.append(a)
    return rows


def a4_repredict() -> dict:
    """Run-4 one-sided A4 cells: per-fit prediction with lepage_c against the stored one."""
    import spec_s4
    out = {}
    for a in load(R4, "A4"):
        p, v = a["params"], a["values"]
        if p["dict"] == "poly3" or p["te"].startswith("sym"):
            continue
        alpha = p["alpha"]
        gam = alpha / 3.0
        beta = v["beta_te"]
        Z = np.sort(G.lepage_c(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Zc": gam, "beta": beta}))))
        sg = np.sign(np.array(v["fit_Kt"]))
        lam = (np.array(v["N_grid"])[-1] / np.array(v["fit_Nstar"])) ** ((3.0 - alpha) / alpha)
        pred = G.psi_sorted(Z, lam * sg)
        end = np.array(v["fit_end"])
        key = f'oe3_al{alpha}_n{p["n"]}_{p["te"]}'
        out[key] = {"predErr_stored": v["predErr"], "predErr_comp": float(np.median(np.abs(end - pred))),
                    "maxAbsPredChange": float(np.max(np.abs(pred - np.array(v["fit_pred"])))),
                    "n_neg": int((sg < 0).sum()), "eps_te": spec_s4.TE_A[p["te"]]}
    return out


def g_curves() -> dict:
    """Post-hoc, not registered: at N_max both predictions of a collapsing fit saturate at 0, so G2
    and G3 cannot tell N* from N*_eff. Compare the whole median curve instead: at every grid N >= 100,
    the median over fits of AUC_N (stored medCurveK) against the median over fits of the predicted
    Psi((N / N*)^{(a*-a)/a} sgn K_t), with N* and with N*_eff (compensated LePage sum)."""
    out = {}
    for a in load(D, "G5"):
        p, v = a["params"], a["values"]
        alpha, beta = p["alpha"], v["beta_te"]
        gam = alpha / 3.0
        Z = np.sort(G.lepage_c(gam, beta, 200_000, np.random.default_rng(H.seed_of({"Zc": gam, "beta": beta}))))
        Ns = np.array(v["N_grid"], float)
        sg = np.sign(np.array(v["fit_Kt"]))
        sel = Ns >= 100
        obs = np.array(v["medCurveK"])[sel]
        row = {}
        for key, nk in (("Nstar", "fit_Nstar"), ("Nstar_eff", "fit_Nstar_eff")):
            ns = np.array(v[nk])
            pc = np.array([np.median(G.psi_sorted(Z, (N / ns) ** ((3.0 - alpha) / alpha) * sg)) for N in Ns[sel]])
            row[f"maxdiff_{key}"] = float(np.max(np.abs(obs - pc)))
            row[f"meandiff_{key}"] = float(np.mean(np.abs(obs - pc)))
        out[f'{p["role"]}_{p["src"]}_n{p["n"]}_{p["te"]}'] = row
    return out


def main() -> int:
    res = {"F1": [], "F2": [], "F3": [], "G1": [], "G2": [], "G3": []}
    fails = {k: [] for k in res}
    print("== F: limit law of Cor. detwindow (exact AUC, S3-C fits)")
    for a in sorted(load(D, "F5"), key=lambda r: (r["params"]["dict"], r["params"]["n"])):
        p, v = a["params"], a["values"]
        for te in ("sym", "one"):
            tag = f'{p["dict"]} n={p["n"]} {te}'
            ks, md = v[f"{te}_ks_total"], v[f"{te}_mean_diff"]
            med_w = float(np.median(np.abs(np.array(v[f"{te}_fit_auc"]) - np.array(v[f"{te}_fit_cond"]))))
            print(f'{tag:22s} N_n={v["N_n"]:.3g} beta={v[f"{te}_beta_te"]:+.0f} KS total {ks:.3f} '
                  f'(window {v[f"{te}_ks_window"]:.3f}, training {v[f"{te}_ks_training"]:.3f}) '
                  f'mean {v[f"{te}_mean_auc"]:.3f} vs {v[f"{te}_mean_lim"]:.3f} (diff {md:+.3f}) '
                  f'median|auc-cond| {med_w:.4f} nan {v[f"{te}_nan"]} | KS H_n,H {v["ks_Hn_H"]:.3f} pos {v["pos"]:.2f}')
            res["F3"].append({"cell": tag, "med_abs_window": med_w})
            if not med_w <= 0.01 or v[f"{te}_nan"] > 0:
                fails["F3"].append(tag)
            if p["n"] == 512000:
                res["F1"].append({"cell": tag, "ks": ks})
                res["F2"].append({"cell": tag, "mean_diff": md})
                if not ks <= 0.10:
                    fails["F1"].append(tag)
                if not abs(md) <= 0.03:
                    fails["F2"].append(tag)
    print("== G: (x, |x|^3), one-sided splits with collapsing fits")
    for a in sorted(load(D, "G5"), key=lambda r: (r["params"]["role"], r["params"]["src"], r["params"]["te"])):
        p, v = a["params"], a["values"]
        tag = f'{p["role"]} {p["src"]} al={p["alpha"]} n={p["n"]} {p["te"]}'
        print(f'{tag:34s} coll {v["share_collapse"]:.2f} ({v["n_collapse"]}) N* {v["coll_medNstar"]:.3g} '
              f'N*eff {v["coll_medNstar_eff"]:.3g} | peak {v["coll_medPeak"]:.3f} end {v["coll_medAucEnd"]:.3f} '
              f'{v["coll_medAucEnd_ci"]} drop {v["coll_medDrop"]:.3f} | err {v["coll_predErr"]:.3f} '
              f'{v["coll_predErr_ci"]} err_eff {v["coll_predErr_eff"]:.3f} {v["coll_predErr_eff_ci"]} | '
              f'W {v["coll_medAucWEnd"]:.3f} noncoll end {v["noncoll_medAucEnd"]:.3f} | '
              f'uncomp err (all fits) {v["predErr_uncomp"]:.3f} vs {v["predErr"]:.3f} | pos {v["pos"]:.2f}')
        row = {"cell": tag, "end": v["coll_medAucEnd"], "drop": v["coll_medDrop"], "err": v["coll_predErr"],
               "err_eff": v["coll_predErr_eff"], "share": v["share_collapse"]}
        if p["role"] == "collapse":
            res["G1"].append(row)
            res["G2"].append(row)
            if not (v["coll_medAucEnd"] < 0.5 and v["coll_medDrop"] >= 0.25):
                fails["G1"].append(tag)
            if not v["coll_predErr"] <= 0.10:
                fails["G2"].append(tag)
        res["G3"].append(row)
        if not v["coll_predErr_eff"] <= 0.10:
            fails["G3"].append(tag)
    print("== verdicts")
    verdicts = {}
    for k in res:
        n = len(res[k])
        ok = n - len(fails[k])
        verdicts[k] = {"pass": len(fails[k]) == 0 and n > 0, "passed": ok, "total": n, "failed": fails[k]}
        print(f'{k}: {"PASS" if verdicts[k]["pass"] else "FAIL"} {ok}/{n}' + (f'  failed: {fails[k]}' if fails[k] else ""))
    print("== reported: run-4 one-sided A4 cells re-predicted with lepage_c")
    a4 = a4_repredict()
    for k, v in a4.items():
        print(f'{k:28s} stored err {v["predErr_stored"]:.4f} compensated err {v["predErr_comp"]:.4f} '
              f'max |pred change| {v["maxAbsPredChange"]:.3f} fits with K_t<0: {v["n_neg"]}')
    print("== post-hoc, not registered: median AUC_N curve (N >= 100) against the median predicted curve")
    gc = g_curves()
    for k, v in gc.items():
        print(f'{k:30s} N*: max {v["maxdiff_Nstar"]:.3f} mean {v["meandiff_Nstar"]:.3f} | '
              f'N*_eff: max {v["maxdiff_Nstar_eff"]:.3f} mean {v["meandiff_Nstar_eff"]:.3f}')
    OUT.write_text(json.dumps({"run": RUN, "verdicts": verdicts, "rows": res, "a4_repredict": a4,
                               "g_curves_posthoc": gc}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
