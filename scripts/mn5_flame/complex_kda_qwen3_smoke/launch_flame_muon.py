# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os
import runpy
import time
from pathlib import Path

import torch
import torch.distributed as dist
from muon import MuonWithAuxAdam
from torchtitan.components import checkpoint as torchtitan_checkpoint
from torchtitan.components import optimizer as torchtitan_optimizer
from torchtitan.tools.logging import logger


def build_muon_optimizers(model_parts, job_config, ft_manager):
    if job_config.optimizer.early_step_in_backward:
        raise NotImplementedError("Muon does not support early optimizer steps in this launcher")
    if ft_manager.enabled:
        raise NotImplementedError("Muon does not support TorchFT in this launcher")

    muon_parameters = {
        id(parameter)
        for model in model_parts
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.ndim == 2
        and "embeddings" not in name
        and "lm_head" not in name
    }

    def optimizer_factory(parameters, **_):
        parameters = list(parameters)
        hidden_weights = [parameter for parameter in parameters if id(parameter) in muon_parameters]
        auxiliary_parameters = [parameter for parameter in parameters if id(parameter) not in muon_parameters]
        return MuonWithAuxAdam(
            [
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": job_config.optimizer.lr,
                    "momentum": job_config.optimizer.momentum,
                    "weight_decay": job_config.optimizer.weight_decay,
                },
                {
                    "params": auxiliary_parameters,
                    "use_muon": False,
                    "lr": job_config.optimizer.adamw_lr,
                    "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
                    "eps": job_config.optimizer.eps,
                    "weight_decay": job_config.optimizer.weight_decay,
                },
            ]
        )

    muon_count = sum(
        parameter.numel()
        for model in model_parts
        for parameter in model.parameters()
        if id(parameter) in muon_parameters
    )
    total_count = sum(parameter.numel() for model in model_parts for parameter in model.parameters())
    logger.info(
        "Using Muon for %s parameters and auxiliary AdamW for %s parameters",
        f"{muon_count:,}",
        f"{total_count - muon_count:,}",
    )
    return torchtitan_optimizer.OptimizersContainer(model_parts, optimizer_factory, {})


def install_model_snapshot_hook() -> None:
    interval = int(os.environ.get("FLAME_MODEL_SNAPSHOT_INTERVAL", "0"))
    if interval <= 0:
        return

    tokens_per_step = int(os.environ["FLAME_TOKENS_PER_STEP"])
    snapshot_root = Path(os.environ["FLAME_MODEL_SNAPSHOT_DIR"])
    original_save = torchtitan_checkpoint.CheckpointManager.save

    @torch.no_grad()
    def save_with_model_snapshot(self, curr_step: int, force: bool = False) -> None:
        original_save(self, curr_step, force)
        if dist.get_rank() != 0 or curr_step == 1 or not (force or curr_step % interval == 0):
            return

        begin = time.monotonic()
        state = {
            name: tensor.detach().to(device="cpu", dtype=torch.bfloat16)
            if torch.is_floating_point(tensor)
            else tensor.detach().cpu()
            for name, tensor in self.states[torchtitan_checkpoint.MODEL].state_dict().items()
        }
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / f"step-{curr_step:09d}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(
            {
                "step": torch.tensor(curr_step),
                "tokens": torch.tensor(curr_step * tokens_per_step),
                "model": state,
            },
            temporary,
        )
        temporary.replace(destination)
        logger.info("Saved bfloat16 model snapshot to %s in %.2f seconds", destination, time.monotonic() - begin)

    torchtitan_checkpoint.CheckpointManager.save = save_with_model_snapshot


def main():
    torchtitan_optimizer.build_optimizers = build_muon_optimizers
    install_model_snapshot_hook()
    runpy.run_module("flame.train", run_name="__main__")


if __name__ == "__main__":
    main()
