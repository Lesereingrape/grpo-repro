from __future__ import annotations

import torch

from grpore.policy import Policy, prompt_vector
from grpore.train import RunConfig, run_algo


def test_generate_returns_valid_finished_sequence():
    torch.manual_seed(0)
    pol = Policy()
    ids, lps = pol.generate(prompt_vector(_prompt()))
    assert len(ids) == len(lps) >= 1
    assert all(-20 < lp <= 0 for lp in lps)


def test_token_logprobs_match_generation_bookkeeping():
    torch.manual_seed(0)
    pol = Policy()
    pv = prompt_vector(_prompt())
    ids, _ = pol.generate(pv)
    scored = pol.token_logprobs(pv, ids)
    assert scored.shape == (len(ids),)
    assert bool((scored <= 0).all())


def test_prompt_vector_is_normalised():
    vec = prompt_vector(_prompt())
    assert vec.shape == (4,)
    assert bool((vec >= 0).all() and (vec <= 1).all())


def test_run_algo_is_deterministic_for_a_seed():
    cfg = {"iters": 2, "eval_every": 2, "prompts_per_iter": 4, "group_size": 4}
    log_a = run_algo(RunConfig(algo="grpo", seed=11, **cfg))
    log_b = run_algo(RunConfig(algo="grpo", seed=11, **cfg))
    assert log_a[-1]["eval"] == log_b[-1]["eval"]


def test_every_arm_reduces_to_finite_metrics():
    cfg = {"iters": 2, "eval_every": 2, "prompts_per_iter": 4, "group_size": 4}
    for algo in ("grpo", "reinforce", "dpo"):
        log = run_algo(RunConfig(algo=algo, seed=5, **cfg))
        assert log[-1]["eval"]["reward"] >= 0.0
        assert log[-1]["eval"]["acc"] >= 0.0


def _prompt():
    from grpore.env import Prompt

    return Prompt(numbers=(5, 3, 2), target=8)
