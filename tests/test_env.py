from __future__ import annotations

from grpore.env import DIGITS, EOS, OPS, Prompt, evaluate, reward, sample_prompts


def test_evaluator_basic_arithmetic():
    # (5 + 3) with digits (5, 3, x): d1 d2 add eos
    p = Prompt(numbers=(5, 3, 2), target=8)
    toks = ["d1", "d2", "add", EOS]
    assert evaluate(toks, p.numbers) == (8.0, True)
    assert reward(toks, p) == 1.0


def test_evaluator_rejects_underflow():
    p = Prompt(numbers=(5, 3, 2), target=8)
    # operator with empty stack
    assert evaluate(["add", EOS], p.numbers) == (None, False)
    assert reward(["add", EOS], p) == 0.0


def test_evaluator_requires_eos_and_single_value():
    p = Prompt(numbers=(5, 3, 2), target=8)
    # no EOS
    assert evaluate(["d1", "d2", "add"], p.numbers)[1] is False
    # two values left on the stack
    assert evaluate(["d1", "d2", EOS], p.numbers)[1] is False


def test_evaluator_draw_limit():
    p = Prompt(numbers=(5, 3, 2), target=8)
    # four digit pushes > three available digits
    assert evaluate(["d1", "d2", "d3", "d1", "add", "add", "add", EOS], p.numbers) == (None, False)


def test_reward_partial_for_valid_miss():
    p = Prompt(numbers=(5, 3, 2), target=8)
    # 5 * 3 = 15, not the target, but combines digits -> 0.1
    assert reward(["d1", "d2", "mul", EOS], p) == 0.1


def test_reward_penalises_trivial_single_digit():
    p = Prompt(numbers=(5, 3, 2), target=5)
    # pushes d1 (== target) with no operator: valid value but trivial -> 0.0
    assert reward(["d1", EOS], p) == 0.0


def test_sample_prompts_are_reproducible():
    a = [p.text for p in sample_prompts(20, seed=7)]
    b = [p.text for p in sample_prompts(20, seed=7)]
    assert a == b


def test_sampled_targets_in_range():
    for p in sample_prompts(200, seed=3):
        assert 1 <= p.target <= 99
        assert all(1 <= n <= 9 for n in p.numbers)


def test_vocab_is_consistent():
    assert set(DIGITS) | set(OPS) | {EOS} == {"d1", "d2", "d3", "add", "sub", "mul", "eos"}
