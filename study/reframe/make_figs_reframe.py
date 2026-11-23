"""Figures of the reframe manuscript, from the hashed artifacts only (no recomputation).

Writes paper/reframe/figs/{fig_s2_curves,fig_lock,fig_rfly}.pdf.
Usage: python make_figs_reframe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
R1 = HERE / "results" / "h1v4-2026-11-22"
R2 = HERE / "results" / "h1v4-s2-2026-11-22"
FIGS = HERE.parent.parent / "paper" / "reframe" / "figs"
OKABE = {"oo": "#0072B2", "ee": "#D55E00", "oe": "#009E73", "eo": "#CC79A7"}
NS_A = (500, 2000, 8000, 32000, 128000)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def cells(run: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(run.glob("*.json")) if f.name != "RUN.json"]


def fig_s2_curves(X: list[dict]) -> None:
    sel = {c["params"]["dict"]: c for c in X if c["params"]["src"] == "gauss" and c["params"]["cont"] == "pareto"
           and c["params"]["te"] == "sym" and c["params"]["n"] == 200}
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    for ax, d in zip(axs, ("oo", "ee", "oe", "eo")):
        v = sel[d]["values"]
        N = v["N_grid"]
        ax.semilogx(N, v["medCurveK"], color=OKABE[d], lw=1.4, label=r"$\hat K$")
        ax.semilogx(N, v["medCurveKW"], color="0.3", lw=1.0, ls="--", label="winsorized")
        ax.semilogx(N, v["medCurveKS"], color="0.55", lw=1.0, ls=":", label="sub-dictionary")
        ax.axvline(v["lambda_m_pow"] * v["medNstar"], color=OKABE[d], lw=0.7, ls="-.")
        ax.axhline(0.5, color="0.8", lw=0.6)
        ax.set_title(r"$\mathsf{" + d + "}$", fontsize=8)
        ax.set_xlabel("window length $N$")
        ax.set_ylim(0.4, 1.02)
    axs[0].set_ylabel(r"median AUC$_N$")
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "fig_s2_curves.pdf")
    plt.close(fig)


def fig_lock(A: list[dict]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.2))
    for c in A:
        p, v = c["params"], c["values"]
        if v["same_parity"] == 1.0:
            axs[0].semilogx(NS_A, [v[f"lock_n{n}"] for n in NS_A], color=OKABE[p["dict"]], lw=0.8, alpha=0.8)
        elif p["pplus"] == 0.5:
            axs[1].semilogx(NS_A, [v[f"pos_n{n}"] for n in NS_A], color=OKABE[p["dict"]], lw=0.8, alpha=0.8,
                            ls="-" if p["astar"] == 4.0 else "--")
    axs[0].set_ylabel("lock frequency")
    axs[0].set_title(r"same parity ($\mathsf{oo}$, $\mathsf{ee}$), 24 cells", fontsize=8)
    axs[1].axhspan(0.5 - 0.087, 0.5 + 0.087, color="0.9")
    axs[1].set_ylabel(r"$\hat P\{\hat K_t > 0\}$")
    axs[1].set_title(r"opposite parity ($\mathsf{oe}$, $\mathsf{eo}$), $p_+ = 1/2$", fontsize=8)
    for ax in axs:
        ax.set_xlabel("training size $n$ per class")
        ax.set_ylim(-0.02, 1.02)
    handles = [plt.Line2D([], [], color=OKABE[d], lw=1.2, label=r"$\mathsf{" + d + "}$") for d in ("oo", "ee", "oe", "eo")]
    handles += [plt.Line2D([], [], color="0.3", lw=0.8, ls="-", label="$a^* = 4$"),
                plt.Line2D([], [], color="0.3", lw=0.8, ls="--", label="$a^* = 12/5$")]
    axs[1].legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "fig_lock.pdf")
    plt.close(fig)


def fig_rfly(X: list[dict]) -> None:
    sel = {(c["params"]["dict"], c["params"]["te"]): c for c in X
           if c["params"]["src"] == "rfly" and c["params"]["n"] == 200}
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.1), sharey=True)
    for ax, d in zip(axs, ("ee", "oe")):
        for te, ls, lab in (("sym", "-", "symmetric test contamination"), ("one", "--", "one-sided test contamination")):
            v = sel[(d, te)]["values"]
            ax.semilogx(v["N_grid"], v["medCurveK"], color=OKABE[d], lw=1.3, ls=ls, label=lab)
        v = sel[(d, "sym")]["values"]
        ax.semilogx(v["N_grid"], v["medCurveKW"], color="0.3", lw=1.0, ls=":", label="winsorized (symmetric)")
        ax.axhline(0.5, color="0.8", lw=0.6)
        ax.set_title(r"$\mathsf{" + d + "}$, RflyMAD clean laws, $n = 200$", fontsize=8)
        ax.set_xlabel("window length $N$")
    axs[0].set_ylabel(r"median AUC$_N$")
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "fig_rfly.pdf")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    c1, c2 = cells(R1), cells(R2)
    A = [c for c in c1 if c["params"].get("block") == "A"]
    X = [c for c in c2 if c["params"].get("block") == "X"]
    fig_s2_curves(X)
    fig_lock(A)
    fig_rfly(X)
    print("wrote", sorted(p.name for p in FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
