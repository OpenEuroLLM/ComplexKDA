# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot the learned transition geometry of Complex-KDA on S3 and S4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch
import torch.nn.functional as F
from einops import rearrange
from train_wordproblem import Model, cycle_lengths, perm_group

from fla.layers.complex_kda_layer import compute_gate

plt.switch_backend("Agg")


CLASS_ORDER = {
    "s3": ["identity", "transposition", "3-cycle"],
    "s4": ["identity", "transposition", "double transposition", "3-cycle", "4-cycle"],
}
CLASS_COLORS = {
    "identity": "#4C4C4C",
    "transposition": "#D55E00",
    "double transposition": "#CC79A7",
    "3-cycle": "#0072B2",
    "4-cycle": "#009E73",
}
CLASS_DISPLAY = {
    "identity": r"$e$",
    "transposition": r"$(ab)$",
    "double transposition": r"$(ab)(cd)$",
    "3-cycle": r"$(abc)$",
    "4-cycle": r"$(abcd)$",
}


def _read_last_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no completed run in {path}")
    return rows[-1]


def _class_name(task: str, permutation: tuple[int, ...]) -> str:
    lengths = cycle_lengths(permutation)
    if not lengths:
        return "identity"
    if lengths == [2]:
        return "transposition"
    if lengths == [2, 2]:
        return "double transposition"
    if lengths == [3]:
        return "3-cycle"
    if lengths == [4]:
        return "4-cycle"
    raise ValueError(f"unsupported cycle type {lengths} for {task}")


def _load_model(row: dict) -> tuple[Model, list[tuple[int, ...]]]:
    elements, _, _ = perm_group(row["task"])
    model = Model(
        vocab=len(elements),
        d_model=row["d_model"],
        n_heads=row["heads"],
        head_dim=row["head_dim"],
        gate=row["gate"],
        allow_neg_eigval=row["allow_neg_eigval"],
        backend="naive_recurrent",
        gate_init_style=row.get("gate_init_style", "shipped"),
        beta_init_style=row.get("beta_init_style", "standard"),
        vocab_in=len(elements) + 1,
        readout=row.get("readout", "mlp"),
        drop_silu=row.get("drop_silu", False),
    )
    state = torch.load(row["ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, elements


@torch.no_grad()
def _analyze(run_path: Path) -> dict:
    row = _read_last_row(run_path)
    model, elements = _load_model(row)
    _, table, identity = perm_group(row["task"])
    layer = model.layer
    tokens = torch.arange(len(elements))
    hidden = model.emb(tokens)

    key = layer.act(layer.k_proj(hidden))
    key = F.normalize(rearrange(key, "t (h d) -> t h d", d=layer.head_k_dim).float(), dim=-1)
    gate_logits = rearrange(layer.f_proj(hidden), "t (h d) -> t h d", d=layer.head_k_dim)
    sign, log_abs_alpha = compute_gate(
        layer.gate,
        gate_logits,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
    )
    if sign is None:
        raise ValueError(f"{run_path} is not a signed-gate run")
    alpha = sign.float() * log_abs_alpha.exp()
    beta = 2.0 * torch.sigmoid(layer.b_proj(hidden).float())

    eye = torch.eye(layer.head_k_dim)
    householder = eye - beta[..., None, None] * key[..., :, None] * key[..., None, :]
    diagonal = torch.diag_embed(alpha)
    transition = householder @ diagonal
    commutator = torch.linalg.matrix_norm(householder @ diagonal - diagonal @ householder, ord="fro")
    eigvals = torch.linalg.eigvals(transition.double())
    table_tensor = torch.as_tensor(table)
    composition_error = torch.linalg.matrix_norm(
        transition[:, None] @ transition[None, :] - transition[table_tensor],
        ord="fro",
    ) / np.sqrt(layer.head_k_dim)
    identity_error = torch.linalg.matrix_norm(
        transition[identity] - eye,
        ord="fro",
    ) / np.sqrt(layer.head_k_dim)

    positive_energy = (key.square() * (sign > 0)).sum(-1)
    negative_energy = (key.square() * (sign < 0)).sum(-1)
    mixing = 2.0 * torch.sqrt((positive_energy * negative_energy).clamp_min(0.0))
    discriminant = 4.0 * (beta - 1.0) - beta.square() * (negative_energy - positive_energy).square()
    predicted_imag = 0.5 * torch.sqrt(discriminant.clamp_min(0.0))
    observed_imag = eigvals.imag.abs().amax(-1).float()

    spectral_score = observed_imag.mean(0)
    active_head = int(spectral_score.argmax().item())
    pca_curves = []
    for head in range(layer.num_heads):
        centered = key[:, head] - key[:, head].mean(0, keepdim=True)
        variance = torch.linalg.svdvals(centered.double()).square()
        explained = variance / variance.sum().clamp_min(torch.finfo(variance.dtype).eps)
        curve = torch.zeros(layer.head_k_dim, dtype=torch.float64)
        curve[:explained.numel()] = explained.cumsum(0)
        if explained.numel() < layer.head_k_dim:
            curve[explained.numel():] = 1.0
        pca_curves.append(curve)

    class_names = [_class_name(row["task"], element) for element in elements]
    class_rank = {name: i for i, name in enumerate(CLASS_ORDER[row["task"]])}
    order = np.array(sorted(range(len(elements)), key=lambda i: (class_rank[class_names[i]], i)))

    return {
        "row": row,
        "elements": elements,
        "classes": class_names,
        "order": order,
        "key": key.numpy(),
        "alpha": alpha.numpy(),
        "beta": beta.numpy(),
        "eigvals": eigvals.numpy(),
        "commutator": commutator.numpy(),
        "composition_error": composition_error.numpy(),
        "identity_error": identity_error.numpy(),
        "mixing": mixing.numpy(),
        "predicted_imag": predicted_imag.numpy(),
        "observed_imag": observed_imag.numpy(),
        "pca": torch.stack(pca_curves).numpy(),
        "active_head": active_head,
    }


def _style() -> None:
    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "font.size": 11.5,
        "legend.fontsize": 8.5,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.right": True,
    })


def _class_spans(data: dict) -> list[tuple[int, int, str]]:
    ordered_classes = [data["classes"][i] for i in data["order"]]
    spans = []
    start = 0
    for index in range(1, len(ordered_classes) + 1):
        if index == len(ordered_classes) or ordered_classes[index] != ordered_classes[start]:
            spans.append((start, index, ordered_classes[start]))
            start = index
    return spans


def _mark_class_spans(ax: plt.Axes, data: dict) -> None:
    for start, end, name in _class_spans(data):
        ax.axvline(end - 0.5, color="white", linewidth=1.0)
        ax.text(
            0.5 * (start + end - 1),
            1.01,
            CLASS_DISPLAY[name],
            color=CLASS_COLORS[name],
            fontsize=7,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )


def _plot_overview(results: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(len(results), 4, figsize=(13.2, 6.4), constrained_layout=True)
    axes = np.atleast_2d(axes)
    beta_image = None
    for row_index, data in enumerate(results):
        task = data["row"]["task"].upper()
        active = data["active_head"]
        order = data["order"]

        ax = axes[row_index, 0]
        beta_image = ax.imshow(data["beta"][order].T, vmin=0, vmax=2, cmap="viridis", aspect="auto")
        ax.axhline(active - 0.5, color="white", linewidth=1.2)
        ax.axhline(active + 0.5, color="white", linewidth=1.2)
        _mark_class_spans(ax, data)
        ax.set_ylabel(f"{task}\nhead")
        ax.set_xlabel("group element (ordered by class)")
        ax.set_yticks(range(data["beta"].shape[1]))
        ax.set_title(r"Householder rate $\beta$", pad=18)

        ax = axes[row_index, 1]
        heads = np.arange(data["beta"].shape[1])
        frac_alpha_neg = (data["alpha"] < 0).mean((0, 2))
        frac_beta_reflection = (data["beta"] > 1.9).mean(0)
        mean_mixing = data["mixing"].mean(0)
        ax.plot(heads, frac_alpha_neg, "o-", markersize=3.5, label=r"$P(\alpha<0)$")
        ax.plot(heads, frac_beta_reflection, "s-", markersize=3.5, label=r"$P(\beta>1.9)$")
        ax.plot(heads, mean_mixing, "^-", markersize=3.5, label=r"mean $2\sqrt{pq}$")
        ax.axvline(active, color="0.5", linestyle=":", linewidth=1.0)
        ax.set(xlabel="head", ylabel="fraction or normalized strength", ylim=(-0.03, 1.03))
        ax.set_xticks(heads)
        ax.set_title("Signed-gate use and mixing")
        if row_index == 0:
            ax.legend(fontsize=7, ncol=1)

        ax = axes[row_index, 2]
        components = np.arange(1, data["pca"].shape[1] + 1)
        for head, curve in enumerate(data["pca"]):
            if head != active:
                ax.plot(components, curve, color="0.75", linewidth=0.7)
        ax.plot(components, data["pca"][active], color="#0072B2", linewidth=2.0, label=f"head {active}")
        ax.axhline(0.95, color="0.35", linestyle=":", linewidth=1.0)
        ax.set(xlabel="number of principal components", ylabel="cumulative explained variance")
        ax.set_xlim(1, data["pca"].shape[1])
        ax.set_ylim(0, 1.02)
        ax.set_title("Token-conditioned key subspace")
        ax.legend(fontsize=8)

        ax = axes[row_index, 3]
        theta = np.linspace(0, 2 * np.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), color="0.65", linestyle="--", linewidth=0.8)
        for name in CLASS_ORDER[data["row"]["task"]]:
            token_mask = np.array([value == name for value in data["classes"]])
            values = data["eigvals"][token_mask, active].reshape(-1)
            ax.scatter(values.real, values.imag, s=10, alpha=0.62, color=CLASS_COLORS[name], label=name)
        ax.axhline(0, color="0.8", linewidth=0.6)
        ax.axvline(0, color="0.8", linewidth=0.6)
        ax.set(xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Full transition spectrum, head {active}")
        if row_index == 0:
            ax.legend(fontsize=6.5, loc="lower left")

    colorbar = fig.colorbar(beta_image, ax=axes[:, 0], shrink=0.76, pad=0.02)
    colorbar.set_label(r"$\beta$")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_mechanism(results: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(len(results), 3, figsize=(11.0, 6.2), constrained_layout=True)
    axes = np.atleast_2d(axes)
    alpha_image = None
    for row_index, data in enumerate(results):
        task = data["row"]["task"]
        active = data["active_head"]
        order = data["order"]

        ax = axes[row_index, 0]
        alpha_image = ax.imshow(
            data["alpha"][order, active].T,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            aspect="auto",
        )
        _mark_class_spans(ax, data)
        ax.set(xlabel="group element (ordered by class)", ylabel=f"{task.upper()}, head {active}\ngate coordinate")
        ax.set_title(r"Learned diagonal gate $\alpha$", pad=18)

        ax = axes[row_index, 1]
        for name in CLASS_ORDER[task]:
            mask = np.array([value == name for value in data["classes"]])
            ax.scatter(
                data["predicted_imag"][mask, active],
                data["observed_imag"][mask, active],
                s=25,
                alpha=0.8,
                color=CLASS_COLORS[name],
                label=name,
            )
        limit = max(
            0.05,
            float(data["predicted_imag"][:, active].max()),
            float(data["observed_imag"][:, active].max()),
        ) * 1.06
        ax.plot([0, limit], [0, limit], color="0.35", linestyle="--", linewidth=0.9)
        ax.set(
            xlabel=r"predicted $|\operatorname{Im}\lambda|$" + "\nfrom sign/key mixing",
            ylabel=r"observed $|\operatorname{Im}\lambda|$" + "\nof full transition",
            xlim=(-0.02 * limit, limit),
            ylim=(-0.02 * limit, limit),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(r"Does $D_{\mathrm{sign}}$–$k$ mixing explain rotation?")
        if row_index == 0:
            ax.legend(fontsize=6.5)

        ax = axes[row_index, 2]
        beta_fraction = data["beta"][order, active] / 2.0
        mean_abs_alpha = np.abs(data["alpha"][order, active]).mean(-1)
        min_abs_alpha = np.abs(data["alpha"][order, active]).min(-1)
        positions = np.arange(len(order))
        ax.plot(positions, beta_fraction, "o-", markersize=3.5, label=r"$\beta/2$")
        ax.plot(positions, mean_abs_alpha, "s-", markersize=3.5, label=r"mean $|\alpha|$")
        ax.plot(positions, min_abs_alpha, "^-", markersize=3.5, label=r"min $|\alpha|$")
        for _, end, _ in _class_spans(data):
            ax.axvline(end - 0.5, color="0.85", linewidth=0.8)
        ax.set(xlabel="group element (ordered by class)", ylabel="fraction of reflection limit", ylim=(-0.03, 1.03))
        ax.set_title("Approach to signed-Householder limit")
        if row_index == 0:
            ax.legend(fontsize=7)

    colorbar = fig.colorbar(alpha_image, ax=axes[:, 0], shrink=0.76, pad=0.02)
    colorbar.set_label(r"$\alpha$")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_s3_concise(data: dict, output: Path) -> None:
    if data["row"]["task"] != "s3":
        raise ValueError("the concise figure is defined for S3 only")

    active = data["active_head"]
    order = data["order"]
    fig = plt.figure(figsize=(9.2, 1.85), constrained_layout=True)
    outer_grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 0.72, 1.25), wspace=0.12)
    beta_grid = outer_grid[0].subgridspec(1, 2, width_ratios=(1.0, 0.05), wspace=0.025)
    alpha_grid = outer_grid[1].subgridspec(1, 2, width_ratios=(1.0, 0.05), wspace=0.025)
    spectrum_grid = outer_grid[3].subgridspec(1, 2, width_ratios=(1.0, 0.42), wspace=0)

    ax_alpha = fig.add_subplot(alpha_grid[0])
    alpha_colorbar_ax = fig.add_subplot(alpha_grid[1])
    alpha_image = ax_alpha.imshow(
        data["alpha"][order, active].T,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    _mark_class_spans(ax_alpha, data)
    ax_alpha.set(xlabel="Group element", ylabel="Gate coordinate")
    ax_alpha.set_xticks([])
    ax_alpha.set_yticks([0, 5, 10, 15])
    ax_alpha.set_title(rf"(b) Gate $\alpha$, head {active}", loc="left", pad=10)
    colorbar = fig.colorbar(alpha_image, cax=alpha_colorbar_ax)
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.set_label(r"$\alpha$", labelpad=0)
    colorbar.set_ticks([-1, 0, 1])

    ax_beta = fig.add_subplot(beta_grid[0])
    beta_colorbar_ax = fig.add_subplot(beta_grid[1])
    beta_image = ax_beta.imshow(
        data["beta"][order].T,
        vmin=0,
        vmax=2,
        cmap="viridis",
        aspect="auto",
    )
    ax_beta.axhline(active - 0.5, color="#B03A2E", lw=1.8)
    ax_beta.axhline(active + 0.5, color="#B03A2E", lw=1.8)
    _mark_class_spans(ax_beta, data)
    ax_beta.set(xlabel="Group element", ylabel="Head")
    ax_beta.set_xticks([])
    ax_beta.set_yticks([0, 3, 7, 11])
    ax_beta.set_yticklabels(["0", "3", "7", str(active)])
    ax_beta.set_title(r"(a) Rate $\beta$ by head", loc="left", pad=10)
    colorbar = fig.colorbar(beta_image, cax=beta_colorbar_ax)
    colorbar.set_label(r"$\beta$")
    colorbar.set_ticks([0, 1, 2])

    ax = fig.add_subplot(outer_grid[2])
    num_components = min(len(data["elements"]) - 1, data["pca"].shape[1])
    components = np.arange(1, num_components + 1)
    explained_variance = data["pca"][active, :num_components]
    ax.plot(components, explained_variance, "o-", color="#0072B2", lw=1.4, ms=3.5)
    ax.axhline(0.95, color="0.45", ls="--", lw=0.9)
    ax.text(num_components - 0.1, 0.93, "95%", color="0.35", fontsize=7, ha="right", va="top")
    ax.set(
        xlabel="Components",
        ylabel="Cumulative variance",
        xlim=(0.8, num_components + 0.2),
        ylim=(0, 1.02),
        xticks=components,
        yticks=[0, 0.5, 1],
    )
    ax.set_title(rf"(c) Key PCA, head {active}", loc="left", pad=10)

    ax = fig.add_subplot(spectrum_grid[0])
    legend_ax = fig.add_subplot(spectrum_grid[1])
    legend_ax.axis("off")
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), color="0.55", ls="--", lw=0.9)
    for name in CLASS_ORDER["s3"]:
        token_mask = np.array([value == name for value in data["classes"]])
        values = data["eigvals"][token_mask, active].reshape(-1)
        ax.scatter(values.real, values.imag, s=14, alpha=0.72, color=CLASS_COLORS[name], label=CLASS_DISPLAY[name])
    target = np.exp(2j * np.pi * np.array([-1.0, 1.0]) / 3.0)
    ax.scatter(
        target.real,
        target.imag,
        s=24,
        facecolors="none",
        edgecolors=CLASS_COLORS["3-cycle"],
        linewidths=0.8,
        alpha=0.8,
        marker="D",
        label=r"target $e^{\pm 2\pi i/3}$",
        zorder=4,
    )
    ax.axhline(0, color="0.82", lw=0.7)
    ax.axvline(0, color="0.82", lw=0.7)
    ax.set(xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", lw=0.6, color="0.88", zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(-0.4, 0.5),
        borderaxespad=0,
        frameon=False,
        handletextpad=0.4,
        ncol=1,
    )
    ax.set_title(rf"(d) Spectrum, head {active}", loc="left", pad=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _s4_targets() -> dict[str, np.ndarray]:
    omega = np.exp(2j * np.pi / 3.0)
    return {
        "identity": np.array([1.0]),
        "transposition": np.array([-1.0, 1.0]),
        "double transposition": np.array([-1.0, 1.0]),
        "3-cycle": np.array([1.0, omega, omega.conjugate()]),
        "4-cycle": np.array([-1.0, 1j, -1j]),
    }


def _plot_s4_study(data: dict, output_dir: Path) -> None:
    if data["row"]["task"] != "s4":
        raise ValueError("the S4 study requires an S4 run")

    active = data["active_head"]
    order = data["order"]
    ordered_classes = [data["classes"][index] for index in order]
    figure = plt.figure(figsize=(10.8, 2.25), constrained_layout=True)
    grid = figure.add_gridspec(1, 4, width_ratios=(1.15, 1.0, 0.82, 1.22), wspace=0.10)

    ax = figure.add_subplot(grid[0])
    image = ax.imshow(data["beta"][order].T, vmin=0, vmax=2, cmap="viridis", aspect="auto")
    ax.axhline(active - 0.5, color="#B03A2E", lw=1.5)
    ax.axhline(active + 0.5, color="#B03A2E", lw=1.5)
    _mark_class_spans(ax, data)
    ax.set(xlabel="Group element", ylabel="Head")
    ax.set_xticks([])
    ax.set_yticks(sorted({0, active, 3, 7, 11}))
    ax.set_title(r"(a) Rate $\beta$ by head", loc="left", pad=9)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_label(r"$\beta$")
    colorbar.set_ticks([0, 1, 2])

    ax = figure.add_subplot(grid[1])
    image = ax.imshow(
        data["alpha"][order, active].T,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    _mark_class_spans(ax, data)
    ax.set(xlabel="Group element", ylabel="Gate coordinate")
    ax.set_xticks([])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_title(rf"(b) Gate $\alpha$, head {active}", loc="left", pad=9)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.055, pad=0.025)
    colorbar.set_label(r"$\alpha$")
    colorbar.set_ticks([-1, 0, 1])

    ax = figure.add_subplot(grid[2])
    positions = np.arange(len(order))
    ax.bar(
        positions,
        data["commutator"][order, active],
        width=0.8,
        color=[CLASS_COLORS[name] for name in ordered_classes],
    )
    for _, end, _ in _class_spans(data):
        ax.axvline(end - 0.5, color="0.85", lw=0.7)
    ax.set(xlabel="Group element", ylabel=r"$\Vert[H_\beta,D_\alpha]\Vert_F$")
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    ax.set_title("(c) Non-commutativity", loc="left", pad=9)

    ax = figure.add_subplot(grid[3])
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), color="0.6", ls="--", lw=0.8)
    for name in CLASS_ORDER["s4"]:
        token_mask = np.array([value == name for value in data["classes"]])
        values = data["eigvals"][token_mask, active].reshape(-1)
        ax.scatter(
            values.real,
            values.imag,
            s=12,
            alpha=0.58,
            color=CLASS_COLORS[name],
            label=CLASS_DISPLAY[name],
        )
        targets = _s4_targets()[name]
        ax.scatter(
            targets.real,
            targets.imag,
            s=30,
            facecolors="none",
            edgecolors=CLASS_COLORS[name],
            linewidths=0.9,
            marker="D",
            zorder=5,
        )
    ax.scatter(
        [],
        [],
        s=30,
        facecolors="none",
        edgecolors="0.2",
        linewidths=0.9,
        marker="D",
        label="theory",
    )
    ax.axhline(0, color="0.82", lw=0.7)
    ax.axvline(0, color="0.82", lw=0.7)
    ax.set(xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", lw=0.5, color="0.9", zorder=0)
    ax.set_title(rf"(d) Spectrum, head {active}", loc="left", pad=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.75,
        ncol=6,
        fontsize=6.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mechanism = output_dir / "s4_complex_kda_interpretability"
    figure.savefig(mechanism.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(mechanism.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), constrained_layout=True)
    heads = np.arange(data["composition_error"].shape[-1])
    mean_error = data["composition_error"].mean((0, 1))
    axes[0].bar(heads, mean_error, color="0.62", width=0.78)
    axes[0].bar(active, mean_error[active], color="#0072B2", width=0.78)
    axes[0].axhline(0, color="0.2", lw=0.8)
    axes[0].set(
        xlabel="Head",
        ylabel=r"mean $\Vert T_gT_h-T_{gh}\Vert_F/\sqrt{d}$",
        xticks=[0, 3, 7, 11],
    )
    axes[0].set_title("(a) S4 group-law defect", loc="left", pad=8)

    lengths = np.array(sorted(int(length) for length in data["row"]["acc"]))
    accuracy = np.array([data["row"]["acc"][str(length)] for length in lengths])
    axes[1].plot(lengths, accuracy, color="#0072B2", lw=1.6)
    axes[1].axhline(data["row"]["chance"], color="0.45", ls="--", lw=0.8, label="chance")
    for length in (15, 18):
        axes[1].axvline(length, color="0.72", ls=":" if length == 18 else "--", lw=0.8)
    axes[1].set(xlabel="Sequence length", ylabel="Accuracy", xlim=(1, lengths.max()), ylim=(0, 1.02))
    axes[1].set_title("(b) Length extrapolation", loc="left", pad=8)
    axes[1].legend(frameon=False, loc="upper right")

    failure = output_dir / "s4_complex_kda_failure_analysis"
    figure.savefig(failure.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(failure.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary(results: list[dict]) -> dict:
    summary = {}
    for data in results:
        task = data["row"]["task"]
        active = data["active_head"]
        pca_components = int(np.searchsorted(data["pca"][active], 0.95) + 1)
        eig = data["eigvals"][:, active]
        persistent_pairs = ((eig.imag > 1e-5) & (np.abs(eig) > 0.9)).sum(-1)
        predicted = data["predicted_imag"][:, active]
        observed = data["observed_imag"][:, active]
        spectral_target_error = {}
        if task == "s4":
            for name, targets in _s4_targets().items():
                mask = np.array([value == name for value in data["classes"]])
                errors = []
                for values in data["eigvals"][mask, active]:
                    errors.append(np.abs(values[:, None] - targets[None, :]).min(0).mean())
                spectral_target_error[name] = float(np.mean(errors))
        summary[task] = {
            "checkpoint": data["row"]["ckpt"],
            "accuracy_at_32": data["row"]["acc"]["32"],
            "accuracy_at_128": data["row"]["acc"]["128"],
            "accuracy_at_512": data["row"]["acc"]["512"],
            "gate_flip_rate_on_evaluation_batch": data["row"]["flip_rate"],
            "gate_saturation_on_evaluation_batch": data["row"]["saturated"],
            "active_head": active,
            "fraction_alpha_negative_all_heads": float((data["alpha"] < 0).mean()),
            "fraction_alpha_negative_active_head": float((data["alpha"][:, active] < 0).mean()),
            "mean_abs_alpha_all_heads": float(np.abs(data["alpha"]).mean()),
            "mean_abs_alpha_active_head": float(np.abs(data["alpha"][:, active]).mean()),
            "mean_beta_all_heads": float(data["beta"].mean()),
            "mean_beta_active_head": float(data["beta"][:, active].mean()),
            "fraction_beta_above_1_9_all_heads": float((data["beta"] > 1.9).mean()),
            "fraction_beta_above_1_9_active_head": float((data["beta"][:, active] > 1.9).mean()),
            "mean_gate_key_mixing_active_head": float(data["mixing"][:, active].mean()),
            "mean_commutator_norm_active_head": float(data["commutator"][:, active].mean()),
            "mean_absolute_rotation_prediction_error_active_head": float(np.abs(predicted - observed).mean()),
            "pca_components_for_95_percent_active_head": pca_components,
            "mean_persistent_complex_pairs_active_head": float(persistent_pairs.mean()),
            "max_persistent_complex_pairs_active_head": int(persistent_pairs.max()),
            "mean_composition_error_per_head": data["composition_error"].mean((0, 1)).tolist(),
            "mean_composition_error_active_head": float(data["composition_error"][:, :, active].mean()),
            "mean_identity_error_all_heads": float(data["identity_error"].mean()),
            "identity_error_active_head": float(data["identity_error"][active]),
            "mean_target_to_learned_spectral_error_active_head": spectral_target_error,
        }
    return summary


def _write_arrays(results: list[dict], path: Path) -> None:
    arrays = {}
    names = (
        "order",
        "key",
        "alpha",
        "beta",
        "eigvals",
        "commutator",
        "composition_error",
        "identity_error",
        "mixing",
        "predicted_imag",
        "observed_imag",
        "pca",
    )
    for data in results:
        prefix = data["row"]["task"]
        for name in names:
            arrays[f"{prefix}_{name}"] = np.asarray(data[name])
        arrays[f"{prefix}_classes"] = np.asarray(data["classes"])
        arrays[f"{prefix}_active_head"] = np.asarray(data["active_head"])
        arrays[f"{prefix}_run_row_json"] = np.asarray(json.dumps(data["row"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="completed JSONL runs, normally one S3 run and one S4 run",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--arrays-output", type=Path)
    args = parser.parse_args()

    _style()
    results = [_analyze(path) for path in args.runs]
    results.sort(key=lambda data: data["row"]["task"])
    _plot_overview(results, args.output_dir / "complex_kda_interpretability")
    _plot_mechanism(results, args.output_dir / "complex_kda_rotation_mechanism")
    s3 = next((data for data in results if data["row"]["task"] == "s3"), None)
    if s3 is not None:
        _plot_s3_concise(s3, args.output_dir / "s3_complex_kda_interpretability")
    s4 = next((data for data in results if data["row"]["task"] == "s4"), None)
    if s4 is not None:
        _plot_s4_study(s4, args.output_dir)
    summary_path = args.output_dir / "complex_kda_interpretability_summary.json"
    summary_path.write_text(json.dumps(_summary(results), indent=2) + "\n")
    arrays_path = args.arrays_output or args.output_dir / "complex_kda_interpretability_arrays.npz"
    _write_arrays(results, arrays_path)
    print(f"wrote interpretability figures, {summary_path}, and {arrays_path}")


if __name__ == "__main__":
    main()
