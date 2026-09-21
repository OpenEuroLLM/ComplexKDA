# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot the learned representation and extrapolation of theory-initialized A5 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch
import torch.nn.functional as F
from a5_exact_tracker import binary_icosahedral, f_plus, isomorphism, transitions
from matplotlib.colors import LogNorm
from train_wordproblem import Model, cycle_lengths, evaluate_continuous, perm_group

from fla.layers.complex_kda_layer import compute_gate

plt.switch_backend("Agg")


CLASS_ORDER = ("identity", "order 2", "order 3", "order 5a", "order 5b")
CLASS_DISPLAY = {
    "identity": r"$e$",
    "order 2": r"$2$",
    "order 3": r"$3$",
    "order 5a": r"$5_a$",
    "order 5b": r"$5_b$",
}
CLASS_COLORS = {
    "identity": "#4C4C4C",
    "order 2": "#D55E00",
    "order 3": "#0072B2",
    "order 5a": "#009E73",
    "order 5b": "#CC79A7",
}


def _style() -> None:
    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams.update({
        "axes.linewidth": 0.9,
        "font.size": 9.5,
        "legend.fontsize": 7.5,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
    })


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no completed runs in {path}")
    return rows


def _load_model(row: dict) -> Model:
    if row["task"] != "a5" or row["layer"] != "kda":
        raise ValueError("the interpretability plot expects A5 KDA checkpoints")
    model = Model(
        60,
        row["d_model"],
        row["heads"],
        row["head_dim"],
        row["gate"],
        row["allow_neg_eigval"],
        "naive_recurrent",
        row.get("gate_init_style", "shipped"),
        row.get("beta_init_style", "standard"),
        vocab_in=61,
        readout=row.get("readout", "mlp"),
        readout_hidden=row.get("readout_hidden"),
        drop_silu=row.get("drop_silu", False),
    )
    model.load_state_dict(torch.load(row["ckpt"], map_location="cpu", weights_only=True))
    model.eval()
    return model


def _targets() -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int, list[tuple[int, ...]]]:
    quaternions, quaternion_table, quaternion_identity = binary_icosahedral()
    elements, table, identity = perm_group("a5")
    table = np.asarray(table)
    mapping = isomorphism(table, identity, quaternion_table, quaternion_identity)
    keys = torch.tensor(quaternions[mapping], dtype=torch.float32)
    return keys, f_plus(transitions(keys)), table, identity, elements


def _classify(element: tuple[int, ...], rotation: torch.Tensor) -> str:
    lengths = cycle_lengths(element)
    if not lengths:
        return "identity"
    order = lengths[-1]
    if order == 2:
        return "order 2"
    if order == 3:
        return "order 3"
    if order == 5:
        cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
        angle = torch.acos(cosine).item()
        return "order 5a" if angle < 3.0 * np.pi / 5.0 else "order 5b"
    raise ValueError(f"unexpected A5 cycle lengths: {lengths}")


def _spans(classes: list[str], order: np.ndarray) -> list[tuple[int, int, str]]:
    ordered = [classes[index] for index in order]
    result = []
    start = 0
    for end in range(1, len(ordered) + 1):
        if end == len(ordered) or ordered[end] != ordered[start]:
            result.append((start, end, ordered[start]))
            start = end
    return result


def _mark_classes(ax: plt.Axes, spans: list[tuple[int, int, str]], labels: bool = True) -> None:
    for start, end, name in spans:
        if end < 60:
            ax.axvline(end - 0.5, color="white", linewidth=0.8)
        if labels:
            ax.text(
                0.5 * (start + end - 1),
                1.015,
                CLASS_DISPLAY[name],
                color=CLASS_COLORS[name],
                fontsize=7,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )


@torch.no_grad()
def _analyze(model: Model, head: int) -> dict:
    target_keys, target_rotations, table, _, elements = _targets()
    layer = model.layer
    if layer.head_k_dim != 4:
        raise ValueError("the A5 representation analysis requires head_dim=4")
    if not 0 <= head < layer.num_heads:
        raise ValueError(f"head must be between 0 and {layer.num_heads - 1}")

    hidden = model.emb(torch.arange(60))
    keys = F.normalize(layer.k_proj(hidden).reshape(60, layer.num_heads, 4).float(), dim=-1)
    gate_logits = layer.f_proj(hidden).reshape(60, layer.num_v_heads, 4)
    sign, log_magnitude = compute_gate(
        layer.gate,
        gate_logits,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
    )
    alpha = sign.float() * log_magnitude.exp()
    beta = 2.0 * torch.sigmoid(layer.b_proj(hidden).float())

    eye = torch.eye(4)
    diagonal = torch.diag_embed(alpha)
    householder = eye - beta[..., None, None] * keys[..., :, None] * keys[..., None, :]
    transition = householder @ diagonal
    decoded = f_plus(transition[:, head])

    target_diagonal = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    target_householder = eye - 2.0 * target_keys[..., :, None] * target_keys[..., None, :]
    target_commutator = torch.linalg.matrix_norm(
        target_householder @ target_diagonal - target_diagonal @ target_householder,
        ord="fro",
    )
    commutator = torch.linalg.matrix_norm(
        householder @ diagonal - diagonal @ householder,
        ord="fro",
    )
    composition_error = torch.linalg.matrix_norm(
        decoded[:, None] @ decoded[None] - decoded[torch.tensor(table)],
        ord="fro",
    )

    classes = [_classify(element, target_rotations[index]) for index, element in enumerate(elements)]
    rank = {name: index for index, name in enumerate(CLASS_ORDER)}
    order = np.array(sorted(range(60), key=lambda index: (rank[classes[index]], index)))

    return {
        "alpha": alpha.numpy(),
        "beta": beta.numpy(),
        "classes": classes,
        "commutator": commutator.numpy(),
        "composition_error": composition_error.numpy(),
        "eigvals": torch.linalg.eigvals(decoded.double()).numpy(),
        "head": head,
        "key_alignment": (keys * target_keys[:, None]).sum(-1).abs().numpy(),
        "order": order,
        "rotation_error": torch.linalg.matrix_norm(decoded - target_rotations, ord="fro").numpy(),
        "spans": _spans(classes, order),
        "target_commutator": target_commutator.numpy(),
        "target_eigvals": torch.linalg.eigvals(target_rotations.double()).numpy(),
    }


def _plot(data: dict, output: Path) -> None:
    figure = plt.figure(figsize=(12.2, 2.3), constrained_layout=True)
    grid = figure.add_gridspec(1, 5, width_ratios=(1.2, 1.05, 0.9, 1.08, 1.0), wspace=0.12)
    head = data["head"]

    ax = figure.add_subplot(grid[0])
    image = ax.imshow(data["beta"][data["order"]].T, vmin=0, vmax=2, cmap="viridis", aspect="auto")
    _mark_classes(ax, data["spans"])
    ax.set_title(r"(a) Rate $\beta$ by head", loc="left", pad=8)
    ax.set_xlabel("Group element")
    ax.set_xticks([])
    ax.set_ylabel("Head")
    ax.set_yticks(range(data["beta"].shape[1]))
    colorbar = figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_label(r"$\beta$")
    colorbar.set_ticks([0, 1, 2])

    ax = figure.add_subplot(grid[1])
    image = ax.imshow(
        data["alpha"][data["order"], head].T,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    _mark_classes(ax, data["spans"])
    ax.set_title(rf"(b) Gate $\alpha$, head {head}", loc="left", pad=8)
    ax.set_xlabel("Group element")
    ax.set_xticks([])
    ax.set_ylabel("Gate coordinate")
    ax.set_yticks(range(4), [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_4$"])
    colorbar = figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_label(r"$\alpha$")
    colorbar.set_ticks([-1, 0, 1])

    ax = figure.add_subplot(grid[2])
    for class_index, name in enumerate(CLASS_ORDER):
        mask = np.array([value == name for value in data["classes"]])
        x = class_index + np.linspace(-0.13, 0.13, mask.sum())
        ax.scatter(x, data["commutator"][mask, head], s=12, color=CLASS_COLORS[name], alpha=0.72)
        ax.scatter(
            class_index,
            np.median(data["target_commutator"][mask]),
            s=34,
            marker="D",
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
    ax.set_xticks(range(5), [CLASS_DISPLAY[name] for name in CLASS_ORDER])
    ax.set_ylabel(r"$\|H_gD-DH_g\|_{\mathrm{F}}$")
    ax.set_title("(c) Non-commutativity", loc="left", pad=8)
    ax.set_ylim(bottom=-0.08)

    spectrum_ax = figure.add_subplot(grid[3])
    theta = np.linspace(0, 2 * np.pi, 512)
    spectrum_ax.plot(np.cos(theta), np.sin(theta), color="0.6", linestyle="--", linewidth=0.8)
    for name in CLASS_ORDER:
        mask = np.array([value == name for value in data["classes"]])
        learned = data["eigvals"][mask].reshape(-1)
        target = data["target_eigvals"][mask].reshape(-1)
        spectrum_ax.scatter(
            learned.real,
            learned.imag,
            s=9,
            color=CLASS_COLORS[name],
            alpha=0.62,
            label=CLASS_DISPLAY[name],
        )
        spectrum_ax.scatter(
            target.real,
            target.imag,
            s=23,
            marker="o",
            facecolors="none",
            edgecolors=CLASS_COLORS[name],
            linewidths=0.75,
        )
    spectrum_ax.axhline(0, color="0.82", linewidth=0.6)
    spectrum_ax.axvline(0, color="0.82", linewidth=0.6)
    spectrum_ax.set(
        xlabel=r"Re$(\lambda)$",
        ylabel=r"Im$(\lambda)$",
        xlim=(-1.08, 1.08),
        ylim=(-1.08, 1.08),
    )
    spectrum_ax.set_aspect("equal", adjustable="box")
    spectrum_ax.set_title(f"(d) Spectrum, head {head}", loc="left", pad=8)

    ax = figure.add_subplot(grid[4])
    ordered_error = data["composition_error"][np.ix_(data["order"], data["order"])]
    positive = ordered_error[ordered_error > 0]
    lower = max(float(positive.min()) if positive.size else 1e-9, 1e-9)
    upper = max(float(ordered_error.max()), lower * 10.0)
    image = ax.imshow(ordered_error, norm=LogNorm(vmin=lower, vmax=upper), cmap="magma", aspect="equal")
    for _, end, _ in data["spans"]:
        if end < 60:
            ax.axhline(end - 0.5, color="white", linewidth=0.45, alpha=0.7)
            ax.axvline(end - 0.5, color="white", linewidth=0.45, alpha=0.7)
    ax.set(xlabel=r"$g$", ylabel=r"$h$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(e) Group law $R_gR_h=R_{gh}$", loc="left", pad=8)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_label("Frobenius error")

    handles, labels = spectrum_ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, handletextpad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _curves(models: list[Model], max_length: int, batch_size: int) -> np.ndarray:
    _, table, identity = perm_group("a5")
    table = np.asarray(table)
    curves = []
    for model in models:
        accuracy = evaluate_continuous(
            model,
            table,
            identity,
            max_length,
            torch.device("cpu"),
            999,
            batch=batch_size,
            bos_id=60,
        )
        curves.append([accuracy[length] for length in range(1, max_length + 1)])
    return np.asarray(curves)


def _stored_curves(path: Path, max_length: int) -> np.ndarray:
    rows = _read_rows(path)
    return np.asarray([
        [row["acc"][str(length)] for length in range(1, max_length + 1)]
        for row in rows
    ])


def _write_curves(rows: list[dict], curves: np.ndarray, path: Path) -> None:
    exported = []
    for row, curve in zip(rows, curves, strict=True):
        result = dict(row)
        result["eval_continuous"] = True
        result["acc"] = {
            str(length): round(float(accuracy), 4)
            for length, accuracy in enumerate(curve, start=1)
        }
        exported.append(json.dumps(result))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(exported) + "\n")


def _plot_extrapolation(initialized: np.ndarray, baseline: np.ndarray, output: Path) -> None:
    lengths = np.arange(1, initialized.shape[1] + 1)
    figure, ax = plt.subplots(figsize=(4.0, 2.55), constrained_layout=True)
    for curves, color, label in (
        (initialized, "#0072B2", "theory initialization"),
        (baseline, "#D55E00", "standard initialization"),
    ):
        mean = curves.mean(0)
        ax.fill_between(lengths, curves.min(0), curves.max(0), color=color, alpha=0.16, linewidth=0)
        ax.plot(lengths, mean, color=color, linewidth=1.6, label=label)
    ax.axvline(32, color="0.35", linestyle="--", linewidth=0.9, label="maximum training length")
    ax.axhline(1.0 / 60.0, color="0.55", linestyle=":", linewidth=0.9, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, lengths[-1])
    ax.set_ylim(-0.02, 1.03)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(ticks, [str(tick) for tick in ticks])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"$A_5$ length extrapolation", loc="left")
    ax.legend(frameon=False, loc="center right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _head_summary(data: dict, row: dict) -> dict:
    head = data["head"]
    commutator_by_class = {}
    for name in CLASS_ORDER:
        mask = np.array([value == name for value in data["classes"]])
        commutator_by_class[name] = {
            "mean": float(data["commutator"][mask, head].mean()),
            "theoretical": float(np.median(data["target_commutator"][mask])),
        }
    return {
        "seed": row["seed"],
        "checkpoint": row["ckpt"],
        "head": head,
        "accuracy": row["acc"],
        "fraction_alpha_negative": float((data["alpha"][:, head] < 0).mean()),
        "maximum_alpha_distance_from_unit_magnitude": float(
            np.abs(np.abs(data["alpha"][:, head]) - 1.0).max()
        ),
        "minimum_beta": float(data["beta"][:, head].min()),
        "maximum_beta_distance_from_two": float(np.abs(data["beta"][:, head] - 2.0).max()),
        "minimum_absolute_key_target_alignment": float(data["key_alignment"][:, head].min()),
        "mean_absolute_key_target_alignment": float(data["key_alignment"][:, head].mean()),
        "mean_decoded_rotation_error_in_original_basis": float(data["rotation_error"].mean()),
        "mean_group_composition_error": float(data["composition_error"].mean()),
        "maximum_group_composition_error": float(data["composition_error"].max()),
        "commutator_by_class": commutator_by_class,
    }


def _summary(results: list[dict], rows: list[dict]) -> dict:
    per_seed = [_head_summary(data, row) for data, row in zip(results, rows, strict=True)]
    accuracy_512 = np.array([float(item["accuracy"]["512"]) for item in per_seed])
    composition_error = np.array([item["mean_group_composition_error"] for item in per_seed])
    return {
        "accuracy_at_512_mean": float(accuracy_512.mean()),
        "accuracy_at_512_std": float(accuracy_512.std()),
        "group_composition_error_mean": float(composition_error.mean()),
        "group_composition_error_std": float(composition_error.std()),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("a5_minimal/theory_init_3seed_3k_exact_embedding.jsonl"),
    )
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, help="representative seed used in the panels")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/a5_complex_kda_interpretability"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("a5_minimal/standard_init_matched_3seed.jsonl"),
    )
    parser.add_argument(
        "--extrapolation-output",
        type=Path,
        default=Path("figures/a5_length_extrapolation"),
    )
    parser.add_argument("--eval-length", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument(
        "--curve-output",
        type=Path,
        help="optional JSONL export of the continuous per-seed accuracies",
    )
    args = parser.parse_args()

    _style()
    rows = _read_rows(args.runs)
    models = [_load_model(row) for row in rows]
    results = [_analyze(model, args.head) for model in models]
    try:
        representative = next(index for index, row in enumerate(rows) if row["seed"] == args.seed)
    except StopIteration as error:
        raise ValueError(f"seed {args.seed} is not present in {args.runs}") from error
    _plot(results[representative], args.output)
    curves = _curves(models, args.eval_length, args.eval_batch)
    _plot_extrapolation(
        curves,
        _stored_curves(args.baseline, args.eval_length),
        args.extrapolation_output,
    )
    if args.curve_output is not None:
        _write_curves(rows, curves, args.curve_output)
    summary_path = args.output.with_name(args.output.name + "_summary.json")
    summary_path.write_text(json.dumps(_summary(results, rows), indent=2) + "\n")
    print(
        f"wrote {args.output.with_suffix('.pdf')}, {args.output.with_suffix('.png')}, "
        f"{args.extrapolation_output.with_suffix('.pdf')}, "
        f"{args.extrapolation_output.with_suffix('.png')}, and {summary_path}"
    )


if __name__ == "__main__":
    main()
