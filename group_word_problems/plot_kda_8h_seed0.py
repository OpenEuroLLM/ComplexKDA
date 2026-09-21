"""Accuracy-vs-length comparison, 8 heads/batch 256: best-per-length-position
across all completed seeds (per user request), for every arm with at least
one finished seed. Note this takes the max independently at EACH length
across seeds, so the curve is an upper envelope -- not necessarily achieved
end-to-end by any single seed's run.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

RUNS = [
    ("s4_paper/A_kda_bos_8h.jsonl",
     "Signed gate + extended Householder range", "#B03A2E", "-"),
    ("s4_paper/B_kda_bos_8h_unsignedgate.jsonl",
     "Unsigned gate + extended Householder range", "#4A7FA5", "-"),
    ("s4_paper/C_kda_bos_8h_signed_defaulthh.jsonl",
     "Signed gate + default Householder range", "#C9A227", "-"),
    ("s4_paper/D_kda_bos_8h_unsigned_defaulthh.jsonl",
     "Unsigned gate + default Householder range", "#5B8C5A", "-"),
    ("s4_paper/E_old_bos_8h.jsonl",
     "Hand-rolled reference layer, signed + extended", "#3B0F5C", "--"),
    ("s4_paper/E_deltaproduct_2h_extended.jsonl",
     "DeltaProduct (2 Householders), extended range", "#E07B39", "-."),
    ("s4_paper/F_gateddeltanet_extended.jsonl",
     "GatedDeltaNet, extended range", "#2F8F8F", "-."),
]


def main(out="s4_paper/kda_8h_seed0_accuracy_vs_length"):
    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams.update({
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "font.size": 11.5,
    })

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    chance, train_len, max_len = None, None, 0
    for path, label, color, ls in RUNS:
        if not os.path.exists(path):
            print(f"  (skip {path}: not started yet)")
            continue
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            print(f"  (skip {path}: no seeds done yet)")
            continue
        lens = sorted(int(k) for k in rows[0]["acc"])
        ys = np.array([[r["acc"][str(L)] for L in lens] for r in rows])  # [seeds, len]
        best = ys.max(0)  # best performance AT EACH length, across seeds
        n = len(rows)
        ax.plot(lens, best, color=color, lw=2.0, ls=ls, zorder=3,
                label=f"{label} (best of n={n})")
        chance, train_len = rows[0]["chance"], rows[0]["train_len"]
        max_len = max(max_len, lens[-1])

    if chance is None:
        raise SystemExit("no data in any run")

    ax.axhline(chance, color="0.5", ls=":", lw=1.3, zorder=0)
    ax.annotate("chance", xy=(max_len, chance), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=9, color="0.4")
    ax.axvline(train_len, color="k", ls="-.", lw=1.3, zorder=0)
    ax.annotate("train\nlength", xy=(train_len, 0.5), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color="0.3", ha="left", va="center")

    ax.set_xlim(0, max_len * 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Best accuracy across seeds, per length")
    ax.set_title(r"$S_4$ word problem: signed/unsigned gate $\times$ default/extended Householder range")
    ax.grid(True, which="both", ls="--", lw=0.6, color="0.88", zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
