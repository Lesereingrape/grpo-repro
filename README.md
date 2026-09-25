# grpo-repro

> A tiny, fully-verifiable lab for comparing **GRPO**, **REINFORCE**, and **DPO**
> on the same policy, the same environment, and the same reward — so the only
> thing that differs is the *optimization itself*.

Group Relative Policy Optimization (GRPO) is the algorithm behind DeepSeek-R1's
reasoning training: it drops PPO's learned critic and instead normalizes reward
*within a group of rollouts for the same prompt*. That idea is easy to describe
and hard to feel. This repo makes it feelable: a ~15k-parameter GRU policy plays a
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

# Re-render the README table from that JSON (no hand-typed numbers)
python experiments/make_report.py --write
```

`grpore ablate` writes the raw per-seed reward/accuracy curves plus the config and the
environment they were measured under, so the table below is reproducible with one
command — and `--out /tmp/again.json` sends a rerun to a scratch path so you can diff it
against the committed artifact instead of overwriting it.

## Results (measured, not illustrative)

<!-- RESULTS:START -->
*Every figure below is produced by `grpore ablate` on CPU and committed as [`results/ablation.json`](results/ablation.json); this block is rendered by `experiments/make_report.py --write`. All three arms share the same policy (48-unit GRU), the same verifiable env, the same reward and the same 120-iteration budget; only the update rule differs. **Accuracy** is the exact-solve rate on 128 held-out puzzles; **reward** also credits the 0.1 for a well-formed but wrong answer. mean ± sample sd over seeds (0, 1, 2).*

- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu — a rerun inside that environment reproduces the artifact field for field except the wall-clock; elsewhere thread count changes float reduction order and the last digits move

### Final held-out reward, accuracy and cost

| algo | reward (mean ± sd) | acc (mean ± sd) | wall-time / run | rollouts per step |
|------|--------------------|-----------------|----------------:|------------------:|
| **GRPO** | 0.187 ± 0.018 | 0.096 ± 0.020 | 126.8 s | 8 (group) |
| REINFORCE | 0.133 ± 0.037 | 0.039 ± 0.041 | 15.9 s | 1 |
| DPO (offline) | 0.111 ± 0.058 | 0.049 ± 0.027 | 10.8 s | 0 (pre-collected pairs) |

### Seed by seed, because a mean hides things

| algo | final acc per seed | final reward per seed | acc sd | climb finished by |
|------|--------------------|-----------------------|-------:|------------------:|
| **GRPO** | s0: 0.094, s1: 0.078, s2: 0.117 | s0: 0.184, s1: 0.170, s2: 0.205 | 0.020 | iter 70 |
| REINFORCE | s0: 0.086, s1: 0.016, s2: 0.016 | s0: 0.176, s1: 0.112, s2: 0.112 | 0.041 | iter 110 |
| DPO (offline) | s0: 0.023, s1: 0.078, s2: 0.047 | s0: 0.045, s1: 0.155, s2: 0.134 | 0.027 | iter 90 |

### What the data actually says

* **GRPO vs REINFORCE on accuracy:** 0.096 against 0.039, and the gap (0.057) is larger than either arm's seed spread (0.020 / 0.041), so the ordering is real at this budget. Group-relative advantage is doing the work of a critic without one: it lowers the variance of the advantage estimate instead of adding parameters to model it.
* **REINFORCE is 8.0x cheaper per step and much less reliable.** Its best seed reaches reward 0.176 while its worst ends at 0.112 (acc 0.016). A running batch-mean baseline absorbs the average but not the per-prompt spread, so whether an arm escapes the floor depends on which prompts the seed happened to draw.
* **DPO improves least, as an offline method should.** Its mean reward went 0.000 -> 0.111 and its mean final accuracy is 0.049 against GRPO's 0.096. It can only reweight preference pairs the frozen reference policy already produced, so its ceiling is inherited from that policy's own distribution: on a task where the untrained policy almost never solves a puzzle, there are almost no informative pairs to learn from. This is a real negative result about offline preference data, not a bug in the loss.
* **Speed of climb (mean curve reaching 95% of its own end):** GRPO iter 70, REINFORCE iter 110, DPO iter 90. Read this column rather than trusting the phrase 'faster convergence' — an arm that plateaus early on a low reward is not converging faster, it is stuck sooner.
* **What this does not show:** 3 seeds, one reward shape, a 48-unit GRU on 3-digit arithmetic. Absolute accuracies are single-digit percentages *by design*; the point is that three update rules were run on identical footing and the differences above survive the per-seed numbers printed next to them.
<!-- RESULTS:END -->

## How the arms differ

```
                 advantage source          stability trick        exploration
GRPO       group mean / group std      PPO clip + KL(β) to ref   online, G=8 rollouts
REINFORCE  running batch-mean baseline none                       online, 1 rollout
DPO        (none — preference pairs)   fixed offline dataset      none
```

The whole comparison lives in ~560 lines across the four modules below:
[`adv.py`](src/grpore/adv.py) (estimators) ·
[`train.py`](src/grpore/train.py) (three loops + metrics) ·
[`policy.py`](src/grpore/policy.py) (the GRU) ·
[`env.py`](src/grpore/env.py) (the verifiable game).

## Tests

```bash
pytest -q          # evaluator, advantages, policy, the three arms, and the
                   # README/artifact integrity guards
```

## Scope & honesty

This is a *pedagogical reproduction harness*, not a state-of-the-art result. The
policy is a GRU, the task is arithmetic on three digits, and accuracy is single-
digit-percentages because the model is deliberately tiny and training is CPU-short.
The value is that the **three algorithms run on identical footing** and the
numbers in the table above are exactly what the code prints — no hidden tuning,
no paper numbers pasted in.

Two things keep that sentence honest rather than aspirational. The Results block is
generated from the committed artifact (`tests/test_readme_matches_results.py` compares
them byte for byte, and re-derives each table cell from the raw runs so a stale number
cannot survive inside regenerated prose), and the artifact itself carries the shared
config, the per-seed curves and the measuring environment
(`tests/test_artifact_is_internally_consistent.py` recomputes every summary field from
those curves). `tests/test_readme_size_claims.py` measures the hand-written
"~15k-parameter" and "~560 lines" figures against the code, because those are the two
numbers the generator cannot fix for you.

A rerun reproduces the artifact field for field except `runtime_sec`, inside the
environment the JSON records — this was checked by writing a second run to a scratch
path (`grpore ablate --out /tmp/again.json`) and diffing it against the committed file.
Wall-clock is the one column that is *not* portable: it moved by a third on the same
machine while other training jobs shared the CPU, which is why it is reported per run
rather than as a speed claim.

## License

MIT
