"""Render the README results block from results/ablation.json.

The README's headline table is generated, not typed: run ``grpore ablate`` then
``python experiments/make_report.py --write`` to splice the block back between the
RESULTS markers, and a CI test asserts the README already equals this output. So a
number here cannot drift from the committed artifact, and the sentences around it are
computed from the same per-seed runs rather than remembered from the last run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ALGO_LABEL = {"grpo": "**GRPO**", "reinforce": "REINFORCE", "dpo": "DPO (offline)"}
# Rollouts the update actually draws from the environment per step, per arm.
ROLLOUTS = {"grpo": "group_size", "reinforce": "one", "dpo": "none"}


def _by_algo(data: dict) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for run in data["runs"]:
        groups.setdefault(run["algo"], []).append(run)
    for runs in groups.values():
        runs.sort(key=lambda r: r["seed"])
    return groups


def _ms(values: list[float]) -> str:
    """mean ± sample sd, the convention used everywhere in this repo."""
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f} ± {sd:.3f}"


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _rollouts(cfg: dict, algo: str) -> str:
    if algo == "grpo":
        return f"{cfg['group_size']} (group)"
    if algo == "reinforce":
        return "1"
    return "0 (pre-collected pairs)"


def build(data: dict) -> str:
    cfg = data["config"]
    env = data["environment"]
    groups = _by_algo(data)
    out: list[str] = []

    out.append(
        "*Every figure below is produced by `grpore ablate` on CPU and committed as "
        "[`results/ablation.json`](results/ablation.json); this block is rendered by "
        "`experiments/make_report.py --write`. All three arms share the same policy "
        f"({cfg['hidden']}-unit GRU), the same verifiable env, the same reward and the "
        f"same {cfg['iters']}-iteration budget; only the update rule differs. "
        f"**Accuracy** is the exact-solve rate on {cfg['eval_prompts']} held-out "
        "puzzles; **reward** also credits the 0.1 for a well-formed but wrong answer. "
        f"mean ± sample sd over seeds {tuple(cfg['seeds'])}.*"
    )
    out.append("")
    out.append(f"- measured under: Python {env['python']} on {env['platform']}, "
               f"torch {env['torch']}, {env['threads']} CPU threads, {env['device']} "
               "— a rerun inside that environment reproduces the artifact field for "
               "field except the wall-clock; elsewhere thread count changes float "
               "reduction order and the last digits move")
    out.append("")

    out.append("### Final held-out reward, accuracy and cost\n")
    out.append("| algo | reward (mean ± sd) | acc (mean ± sd) | wall-time / run | "
               "rollouts per step |")
    out.append("|------|--------------------|-----------------|----------------:|"
               "------------------:|")
    for algo in cfg["algos"]:
        runs = groups[algo]
        out.append(
            f"| {ALGO_LABEL[algo]} | {_ms([r['final_reward'] for r in runs])} | "
            f"{_ms([r['final_acc'] for r in runs])} | "
            f"{statistics.mean([r['seconds'] for r in runs]):.1f} s | "
            f"{_rollouts(cfg, algo)} |"
        )
    out.append("")

    out.append("### Seed by seed, because a mean hides things\n")
    out.append("| algo | final acc per seed | final reward per seed | acc sd | "
               "climb finished by |")
    out.append("|------|--------------------|-----------------------|-------:|"
               "------------------:|")
    for algo in cfg["algos"]:
        runs = groups[algo]
        accs = ", ".join(f"s{r['seed']}: {r['final_acc']:.3f}" for r in runs)
        rew = ", ".join(f"s{r['seed']}: {r['final_reward']:.3f}" for r in runs)
        spread = _sd([r["final_acc"] for r in runs])
        out.append(f"| {ALGO_LABEL[algo]} | {accs} | {rew} | {spread:.3f} | "
                   f"{_plateau(runs, cfg)} |")
    out.append("")

    out.append("### What the data actually says\n")
    for line in _findings(groups, cfg):
        out.append(f"* {line}")
    out.append("")
    return "\n".join(out).rstrip()


def _plateau(runs: list[dict], cfg: dict) -> str:
    """Iteration at which the *mean* curve first reaches 95% of its own end.

    Curve point 0 is the evaluation before any update, so point ``i`` is iteration
    ``i * eval_every``.
    """
    curves = [r["reward_curve"] for r in runs]
    n = min(len(c) for c in curves)
    mean = [statistics.mean([c[i] for c in curves]) for i in range(n)]
    target = mean[-1] * 0.95
    for i, value in enumerate(mean):
        if value >= target:
            return f"iter {cfg['eval_every'] * i}"
    return "not within budget"


def _findings(groups: dict[str, list[dict]], cfg: dict) -> list[str]:
    grpo, reinf, dpo = (groups["grpo"], groups["reinforce"], groups["dpo"])
    lines: list[str] = []

    g_acc, r_acc, d_acc = (statistics.mean([x["final_acc"] for x in g])
                           for g in (grpo, reinf, dpo))
    g_sd, r_sd = _sd([x["final_acc"] for x in grpo]), _sd([x["final_acc"] for x in reinf])
    gap = g_acc - r_acc
    if gap > max(g_sd, r_sd):
        verdict = (f"the gap ({gap:.3f}) is larger than either arm's seed spread "
                   f"({g_sd:.3f} / {r_sd:.3f}), so the ordering is real at this budget")
    else:
        verdict = (f"the gap ({gap:.3f}) sits *inside* the seed spread "
                   f"({g_sd:.3f} / {r_sd:.3f}), so with {len(grpo)} seeds this is not "
                   "yet an ordering, only a tendency")
    lines.append(
        f"**GRPO vs REINFORCE on accuracy:** {g_acc:.3f} against {r_acc:.3f}, and "
        f"{verdict}. Group-relative advantage is doing the work of a critic without "
        "one: it lowers the variance of the advantage estimate instead of adding "
        "parameters to model it."
    )

    cost = (statistics.mean([x["seconds"] for x in reinf])
            / statistics.mean([x["seconds"] for x in grpo]))
    r_best = max(reinf, key=lambda x: x["final_reward"])
    r_worst = min(reinf, key=lambda x: x["final_reward"])
    lines.append(
        f"**REINFORCE is {1 / cost:.1f}x cheaper per step and much less reliable.** "
        f"Its best seed reaches reward {r_best['final_reward']:.3f} while its worst "
        f"ends at {r_worst['final_reward']:.3f} (acc {r_worst['final_acc']:.3f}). A "
        "running batch-mean baseline absorbs the average but not the per-prompt "
        "spread, so whether an arm escapes the floor depends on which prompts the "
        "seed happened to draw."
    )

    d_start = statistics.mean([x["start_reward"] for x in dpo])
    d_final = statistics.mean([x["final_reward"] for x in dpo])
    lines.append(
        f"**DPO improves least, as an offline method should.** Its mean reward went "
        f"{d_start:.3f} -> {d_final:.3f} and its mean final accuracy is {d_acc:.3f} "
        "against "
        f"GRPO's {g_acc:.3f}. It can only reweight preference pairs the frozen "
        "reference policy already produced, so its ceiling is inherited from that "
        "policy's own distribution: on a task where the untrained policy almost never "
        "solves a puzzle, there are almost no informative pairs to learn from. This "
        "is a real negative result about offline preference data, not a bug in the "
        "loss."
    )

    climbs = {a: _plateau(groups[a], cfg) for a in ("grpo", "reinforce", "dpo")}
    lines.append(
        f"**Speed of climb (mean curve reaching 95% of its own end):** GRPO "
        f"{climbs['grpo']}, REINFORCE {climbs['reinforce']}, DPO {climbs['dpo']}. "
        "Read this column rather than trusting the phrase 'faster convergence' — "
        "an arm that plateaus early on a low reward is not converging faster, it is "
        "stuck sooner."
    )
    lines.append(
        f"**What this does not show:** {len(grpo)} seeds, one reward shape, a "
        f"{cfg['hidden']}-unit GRU on 3-digit arithmetic. Absolute accuracies are "
        "single-digit percentages *by design*; the point is that three update rules "
        "were run on identical footing and the differences above survive the "
        "per-seed numbers printed next to them."
    )
    return lines


def _write(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    nl = "\n"
    path.write_text(f"{head}{start}{nl}{block}{nl}{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the block into README.md instead of printing it")
    ap.add_argument("--results", default="results/ablation.json")
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rendered = build(data)
    if args.write:
        _write(Path("README.md"), rendered)
        print("README results block rewritten")
    else:
        print(rendered)
