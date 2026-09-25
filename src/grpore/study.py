"""What ``results/ablation.json`` is: the config, the machine, and the raw runs.

The artifact used to be a bare list of nine per-seed records, which is enough to
rebuild the table but not enough to say *under what* it was measured. CPU float
reduction order follows the thread count and the torch build, so a "reproducible
in CPU seconds" claim without the environment attached is a claim about a machine
nobody can point at. This module owns that provenance so the CLI, the report
renderer and the tests all read the same definition.
"""

from __future__ import annotations

import platform
import sys

import torch

from .train import PUBLISHED_ITERS, RunConfig

#: The seeds the published table averages over.
PUBLISHED_SEEDS = (0, 1, 2)

#: Order the arms appear in the README table.
ALGO_ORDER = ("grpo", "reinforce", "dpo")


def environment() -> dict:
    """Record enough of the machine to make 'bit-exact' mean something."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def published_config(iters: int = PUBLISHED_ITERS) -> dict:
    """The hyperparameters every arm shares, read off the code's own defaults."""
    defaults = RunConfig()
    return {
        "algos": list(ALGO_ORDER),
        "seeds": list(PUBLISHED_SEEDS),
        "iters": iters,
        "prompts_per_iter": defaults.prompts_per_iter,
        "group_size": defaults.group_size,
        "inner_epochs": defaults.inner_epochs,
        "lr": defaults.lr,
        "clip": defaults.clip,
        "beta": defaults.beta,
        "hidden": defaults.hidden,
        "eval_prompts": defaults.eval_prompts,
        "eval_every": defaults.eval_every,
    }


def build_artifact(runs: list[dict], config: dict, runtime_sec: float) -> dict:
    return {
        "environment": environment(),
        "config": config,
        "runtime_sec": round(runtime_sec, 1),
        "runs": runs,
    }
