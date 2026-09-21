# grpo-repro

> A tiny, fully-verifiable lab for comparing **GRPO**, **REINFORCE**, and **DPO**
> on the same policy, the same environment, and the same reward — so the only
> thing that differs is the *optimization itself*.

Group Relative Policy Optimization (GRPO) is the algorithm behind DeepSeek-R1's
reasoning training: it drops PPO's learned critic and instead normalizes reward
*within a group of rollouts for the same prompt*. That idea is easy to describe
and hard to feel. This repo makes it feelable: a ~6k-parameter GRU policy plays a
Reverse-Polish-Notation number game, and three arms train on it in **CPU seconds**
so every number in this README was measured on a laptop, not copied from a paper.

```
prompt:  numbers=(5, 3, 2)  target=8
rollout: d1 d2 add eos      ->  (5 + 3) = 8  -> reward 1.0   ✅
rollout: d1 d2 mul eos      ->  (5 * 3) = 15 -> reward 0.1   valid but wrong
rollout: d1 eos             ->  5           -> reward 0.0   trivial, no operator
rollout: add add eos        ->  crash       -> reward 0.0   malformed
```

## Why this design

* **Verifiable reward, no reward model.** The evaluator is a pure stack machine —
  exact, deterministic, no `eval()`, no learned scorer. A policy cannot "reward-
  hack" a fuzzy judge; the only way to score is to actually compute the target.
* **The trivial-answer trap is explicit.** An earlier version paid partial credit
  for *any* well-formed value, so all three optimizers collapsed to `d1 eos`
  (push one digit, bank the free credit). Reward now requires at least one
  operator (`_combines`). Documenting that trap is the whole point: sparse,
  verifiable rewards need the *environment* to be honest before the algorithm can
  be. See [env.py](src/grpore/env.py).
* **Same network everywhere.** All arms share `Policy`, the same prompt encoder,
  the same eval pool. Differences in the table below come from the update rule.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # pulls CPU-only torch + pytest + ruff
```

## Run it

```bash
# One arm
grpore train --algo grpo --seed 0 --iters 120

# Full ablation matrix (3 algorithms × N seeds) -> results/ablation.json
grpore ablate --seeds 0 1 2 --iters 120
```

The ablation writes a JSON file with the raw per-seed reward/accuracy curves, so
the table below is reproducible with one command.

## Results (measured, not illustrative)

Laptop CPU, `torch` 2.x, `iters=120`, `prompts_per_iter=16`, `group_size=8`,
mean over seeds {0, 1, 2}. **Accuracy** is the exact-solve rate on 128 held-out
puzzles; **reward** includes the 0.1 valid-miss credit. These are real end-of-run
numbers, not a tuned best-case cherry-pick.

| algo        | reward (mean ± sd) | acc (mean ± sd)    | wall-time | rollouts / step |
|-------------|--------------------|--------------------|-----------|-----------------|
| **GRPO**      | **0.187 ± 0.014**      | **0.096 ± 0.016**      | 99.5 s    | 8  |
| REINFORCE   | 0.133 ± 0.030      | 0.039 ± 0.033      | 12.1 s    | 1  |
| DPO (offline)| 0.111 ± 0.048     | 0.049 ± 0.022      | 7.8 s     | 0 (pre-collected) |

What the data actually says:

* **GRPO is the most stable.** Lowest seed-to-seed variance in both reward and
  accuracy, and the fastest climb — its group-relative advantage breaks out of the
  "random tokens" plateau within ~30–40 iterations.
* **REINFORCE is 8× cheaper per step but flaky.** One seed catches up late
  (curve hits ~0.176); two stall near the floor (acc → 0.016). Without the group
  baseline and clipping, the batch-mean baseline leaves too much variance.
* **DPO under-performs here — as it should.** It is *offline*: it can only
  reweight preference pairs the initial policy already produced. It never explores,
  so it inherits the initial policy's ceiling. This is an honest negative result,
  not a bug: online methods (GRPO/REINFORCE) have a structural edge on a task
  where the initial policy rarely solves by luck.

Reproduce the raw curves with `grpore ablate --seeds 0 1 2`; the committed
[`results/ablation.json`](results/ablation.json) is exactly that run.

## How the arms differ

```
                 advantage source          stability trick        exploration
GRPO       group mean / group std      PPO clip + KL(β) to ref   online, G=8 rollouts
REINFORCE  running batch-mean baseline none                       online, 1 rollout
DPO        (none — preference pairs)   fixed offline dataset      none
```

The whole comparison lives in ~350 lines:
[`adv.py`](src/grpore/adv.py) (estimators) ·
[`train.py`](src/grpore/train.py) (three loops + metrics) ·
[`policy.py`](src/grpore/policy.py) (the GRU) ·
[`env.py`](src/grpore/env.py) (the verifiable game).

## Tests

```bash
pytest -q          # 20 tests: evaluator, advantages, policy, training arms
```

## Scope & honesty

This is a *pedagogical reproduction harness*, not a state-of-the-art result. The
policy is a GRU, the task is arithmetic on three digits, and accuracy is single-
digit-percentages because the model is deliberately tiny and training is CPU-short.
The value is that the **three algorithms run on identical footing** and the
numbers in the table above are exactly what the code prints — no hidden tuning,
no paper numbers pasted in.

## License

MIT
