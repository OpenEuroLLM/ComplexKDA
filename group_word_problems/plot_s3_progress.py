"""Accuracy-vs-length, S3, best-per-length across completed seeds -- progress
snapshot while the s3_paper/ sweep is still running. Mirrors plot_kda_8h_seed0.py.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

RUNS = [
    ("s3_paper/A_kda_signed_extended.jsonl",
     "Signed gate + extended Householder range", "#B03A2E", "-"),
    ("s3_paper/B_kda_unsigned_extended.jsonl",
     "Unsigned gate + extended Householder range", "#4A7FA5", "-"),
    ("s3_paper/C_kda_signed_default.jsonl",
     "Signed gate + default Householder range", "#C9A227", "-"),
    ("s3_paper/D_kda_unsigned_default.jsonl",
     "Unsigned gate + default Householder range", "#5B8C5A", "-"),
    ("s3_paper/E_old_signed_extended.jsonl",
     "Hand-rolled reference layer, signed + extended", "#3B0F5C", "--"),
    ("s3_paper/F_deltaproduct_extended.jsonl",
     "DeltaProduct (2 Householders, key silu dropped), extended range", "#E07B39", "-."),
    ("s3_paper/G_gateddeltanet_extended.jsonl",
     "GatedDeltaNet, extended range", "#2F8F8F", "-."),
]


def main(out="s3_paper/s3_accuracy_vs_length_progress"):
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
        best = ys.max(0)
        n = len(rows)
        ax.plot(lens, best, color=color, lw=2.0, ls=ls, zorder=3,
                label=f"{label} (best of n={n}{'*' if n < 3 else ''})")
        chance, train_len = rows[0]["chance"], rows[0]["train_len"]
        max_len = max(max_len, lens[-1])

    if chance is None:
        raise SystemExit("no data in any run yet")

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
    ax.set_title(r"$S_3$ word problem (progress snapshot, *=sweep still running)")
    ax.grid(True, which="both", ls="--", lw=0.6, color="0.88", zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
