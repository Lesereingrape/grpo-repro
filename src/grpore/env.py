"""Verifiable-reward toy environment: RPN number puzzles.

Each prompt shows up to three digits and a target number; the policy emits a
Reverse-Polish-Notation token sequence (push digits / apply operator). The
reward function is a pure, exact evaluator — no string eval, no ambiguity:

    1.0  sequence evaluates to the target
    0.1  sequence combines at least two digits but misses the target
    0.0  malformed sequence, or a trivial one-digit push (no operator)

Verifiable dense-but-sparse rewards are exactly the GRPO training regime
(DeepSeek-R1 style): no learned reward model, so anything the algorithms
score differs purely because of the optimization itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DIGITS = ("d1", "d2", "d3")
OPS = ("add", "sub", "mul")
EOS = "eos"
VOCAB = DIGITS + OPS + (EOS,)
TOK = {t: i for i, t in enumerate(VOCAB)}
ID = {i: t for t, i in TOK.items()}
MAX_STEPS = 10


@dataclass(frozen=True)
class Prompt:
    numbers: tuple[int, int, int]
    target: int

    @property
    def text(self) -> str:
        return f"{self.numbers} -> {self.target}"


def sample_prompts(n: int, seed: int) -> list[Prompt]:
    """Random digit triples; target derived from a solvable random expression
    most of the time, so the task is hard but not impossible."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        nums = tuple(int(rng.randint(1, 9)) for _ in range(3))
        target = _random_expr_value(nums, rng) if rng.random() < 0.8 else rng.randint(1, 30)
        out.append(Prompt(numbers=nums, target=max(1, min(target, 99))))
    return out


def _random_expr_value(nums: tuple[int, ...], rng: random.Random) -> int:
    """Value of a random solvable RPN expression over the given digits."""
    stack = [float(rng.choice(nums))]
    for _ in range(2):
        if len(stack) >= 2 and rng.random() < 0.7:
            b, a = stack.pop(), stack.pop()
            stack.append(_apply(a, b, rng.choice(OPS)))
        else:
            stack.append(float(rng.choice(nums)))
    while len(stack) > 1:
        b, a = stack.pop(), stack.pop()
        stack.append(_apply(a, b, rng.choice(OPS)))
    value = stack[0]
    return int(value) if float(value).is_integer() else 1


def _apply(a: float, b: float, op: str) -> float:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    return a * b


def evaluate(tokens: list[str], numbers: tuple[int, int, int]) -> tuple[float | None, bool]:
    """Return (final_value, well_formed). Stack errors → (None, False)."""
    stack: list[float] = []
    used = 0
    finished = False
    for t in tokens:
        if t == EOS:
            finished = True
            break
        if t in DIGITS:
            idx = int(t[1]) - 1
            stack.append(float(numbers[idx]))
            used += 1
            if used > 3:
                return None, False  # only three digits available
        else:
            if len(stack) < 2:
                return None, False
            b, a = stack.pop(), stack.pop()
            stack.append(_apply(a, b, t))
    if not finished or len(stack) != 1:
        return None, False
    return stack[0], True


def _combines(tokens: list[str]) -> bool:
    """True if the sequence applies at least one operator (a real expression).

    Without this the whole game degenerates: pushing a single digit is a
    "valid" answer, so partial credit pays out for free and every optimizer
    collapses to ``d1 eos``. Requiring a combination is what makes the
    verifiable reward actually verifiable.
    """
    return any(t in OPS for t in tokens)


def reward(tokens: list[str], p: Prompt) -> float:
    value, ok = evaluate(tokens, p.numbers)
    if not ok or not _combines(tokens):
        return 0.0
    if value is not None and float(value).is_integer() and int(value) == p.target:
        return 1.0
    return 0.1
