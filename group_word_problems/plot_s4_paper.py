"""Publication-style accuracy-vs-length plot for the S4 investigation.

Reads the five s4_paper/*.jsonl files (continuous eval, one row per seed,
each row's "acc" a dict of every integer length 1..eval_len) and plots
mean +- std across seeds, log-x, chance/train-length reference lines.
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

ARMS = [
    ("A_unfixed", "Unfixed KDA (shipped)", "#8C8C8C", "--"),
    ("B_wdonly", "Shipped KDA + weight decay", "#4A7FA5", "-."),
    ("C_fixesonly", "Fixed KDA, no weight decay", "#C9A227", "-."),
    ("D_recipe", "Fixed KDA + weight decay (ours)", "#B03A2E", "-"),
    ("E_old", "Hand-rolled reference layer", "#3B0F5C", "-"),
]


def load(tag):
    """Loads raw accuracy, then rescales PER SEED to (acc - chance) / (1 -
    chance), clipped at 0: 0 = chance/random choice, 1 = perfect. Scaling
    before averaging (not after) keeps mean/std internally consistent."""
    path = f"s4_paper/{tag}.jsonl"
    try:
        rows = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        return None
    if not rows:
        return None
    lens = sorted(int(k) for k in rows[0]["acc"])
    chance = rows[0]["chance"]
    raw = np.array([[r["acc"][str(L)] for L in lens] for r in rows])  # [seeds, len]
    ys = np.clip((raw - chance) / (1.0 - chance), 0.0, None)
    return dict(lens=lens, mean=ys.mean(0), std=ys.std(0), n=len(rows),
                chance=chance, train_len=rows[0]["train_len"])


def main(out="s4_paper/s4_accuracy_vs_length"):
    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams.update({
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "font.size": 12,
        "legend.fontsize": 10.5,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ref = None
    for tag, label, color, ls in ARMS:
        d = load(tag)
        if d is None:
            print(f"  (skip {tag}: no data yet)")
            continue
        ref = ref or d
        n = d["n"]
        ax.plot(d["lens"], d["mean"], color=color, ls=ls, lw=2.2,
                label=f"{label}  (n={n})", zorder=3)
        if n > 1:
            ax.fill_between(d["lens"], d["mean"] - d["std"], d["mean"] + d["std"],
                            color=color, alpha=0.15, zorder=1, linewidth=0)

    if ref is None:
        sys.exit("no data in any s4_paper/*.jsonl yet")

    ax.axhline(0.0, color="0.5", ls=":", lw=1.3, zorder=0)
    ax.annotate("chance", xy=(ref["lens"][-1], 0.0), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=9, color="0.4")
    ax.axvline(ref["train_len"], color="k", ls="-.", lw=1.3, zorder=0)
    ax.annotate("train\nlength", xy=(ref["train_len"], 0.5), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color="0.3", ha="left", va="center")

    ax.set_xscale("log", base=2)
    ax.set_xlim(1, ref["lens"][-1] * 1.03)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Scaled accuracy (0 = random choice, 1 = perfect)")
    ax.set_title(r"Length generalisation on $S_4$ word problems")
    ax.grid(True, which="both", ls="--", lw=0.6, color="0.88", zorder=0)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
