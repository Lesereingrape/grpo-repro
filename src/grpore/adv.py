"""Advantage estimators — the one real difference between the arms.

GRPO replaces the learned critic of PPO with a *group-relative* baseline:
for each prompt we roll out G completions, and a completion's advantage is
how much better its verifiable reward is than the group mean, in units of
the group's standard deviation. No value network, no GAE — the baseline is
free because the group is right there.

REINFORCE keeps a running batch-mean baseline instead, which is the classic
policy-gradient variance trick and our control arm.
"""

from __future__ import annotations

import statistics


def group_relative(rewards: list[float]) -> list[float]:
    """Normalize one group's rewards to zero mean / unit std.

    A constant group (all same reward) carries no signal, so its advantages
    are all zero rather than a divide-by-one blowup.
    """
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < 1e-8:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]


def mean_baseline(rewards: list[float], running: float) -> tuple[list[float], float]:
    """REINFORCE advantages against a running-average baseline.

    Returns per-sample advantages and the updated baseline (EMA over batch
    means). Keeping the baseline historical — not the current batch mean —
    is what stops the estimator from being biased by its own samples.
    """
    advs = [r - running for r in rewards]
    batch_mean = statistics.fmean(rewards) if rewards else running
    updated = 0.9 * running + 0.1 * batch_mean
    return advs, updated
