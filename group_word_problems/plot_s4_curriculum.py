"""Accuracy-vs-length for the S4 curriculum/Muon sweep (sweep_curriculum/).

Best-per-length across seeds, as in the earlier figures: with n=3 and one seed
in an arm often solving the task while the others barely fit the training
length, a mean would describe no run that actually happened.

Marks the curriculum's top length (32) rather than --train-len (8): the model
trains directly at every length in --curriculum-lens, so everything up to 32 is
in-distribution and only beyond it is extrapolation.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

RUNS = [
    ("sweep_curriculum/s4_A_kda_signed_extended.jsonl",
     "Signed gate + extended Householder", "#B03A2E", "-"),
    ("sweep_curriculum/s4_C_kda_signed_default.jsonl",
     "Signed gate + default Householder", "#C9A227", "-"),
    ("sweep_curriculum/s4_B_kda_unsigned_extended.jsonl",
     "Unsigned gate + extended Householder", "#4A7FA5", "-"),
    ("sweep_curriculum/s4_D_kda_unsigned_default.jsonl",
     "Unsigned gate + default Householder", "#5B8C5A", "-"),
    ("sweep_curriculum/s4_E_gateddeltanet_extended.jsonl",
     "GatedDeltaNet, extended", "#2F8F8F", "-."),
    ("sweep_curriculum/s4_F_deltaproduct2_extended.jsonl",
     "DeltaProduct (2 Householders), extended", "#E07B39", "-."),
]


def main(out="sweep_curriculum/s4_curriculum_accuracy_vs_length"):
    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams.update({
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "font.size": 11.5,
    })

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    chance, curric_top, max_len = None, None, 0
    for path, label, color, ls in RUNS:
        if not os.path.exists(path):
            print(f"  (skip {path}: not started)")
            continue
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            print(f"  (skip {path}: no seeds done)")
            continue
        lens = sorted(int(k) for k in rows[0]["acc"])
        ys = np.array([[r["acc"][str(L)] for L in lens] for r in rows])
        n = len(rows)
        ax.plot(lens, ys.max(0), color=color, lw=1.9, ls=ls, zorder=3,
                label=f"{label} (best of {n}{'*' if n < 3 else ''})")
        chance = rows[0]["chance"]
        curric_top = max(rows[0]["curriculum_lens"])
        max_len = max(max_len, lens[-1])

    if chance is None:
        raise SystemExit("no data yet")

    ax.axhline(chance, color="0.5", ls=":", lw=1.3, zorder=0)
    ax.annotate("chance", xy=(max_len, chance), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=9, color="0.4")
    ax.axvline(curric_top, color="k", ls="-.", lw=1.3, zorder=0)
    ax.annotate(f"longest length\nseen in training ({curric_top})",
                xy=(curric_top, 0.97), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color="0.3", ha="left", va="top")

    ax.set_xscale("log", base=2)
    ax.set_xlim(1, max_len)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Evaluation sequence length")
    ax.set_ylabel("Best accuracy across seeds, per length")
    ax.set_title(r"$S_4$ word problem — curriculum 4,6,8,16,32 + Muon, 12 heads")
    ax.grid(True, which="major", ls="--", lw=0.6, color="0.88", zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
