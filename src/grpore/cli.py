"""Command-line entry point: run one arm or the full ablation matrix.

    grpore train   --algo grpo --seed 0
    grpore ablate  --seeds 0 1 2 --out results/ablation.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .train import PUBLISHED_ITERS, RunConfig, run_algo, to_jsonl

ALGOS = ("grpo", "reinforce", "dpo")


def _summary(log: list[dict]) -> dict:
    evals = [row["eval"] for row in log if "eval" in row]
    final = evals[-1] if evals else {}
    return {
        "algo": log[0]["algo"] if log else "?",
        "iters": len(log),
        "reward_curve": [round(e["reward"], 4) for e in evals],
        "acc_curve": [round(e["acc"], 4) for e in evals],
        "start_reward": evals[0]["reward"] if evals else None,
        "final_reward": final.get("reward"),
        "final_acc": final.get("acc"),
    }


def cmd_train(args: argparse.Namespace) -> int:
    cfg = RunConfig(algo=args.algo, seed=args.seed, iters=args.iters)
    t0 = time.time()
    log = run_algo(cfg)
    dt = time.time() - t0
    if args.jsonl:
        Path(args.jsonl).write_text(to_jsonl(log), encoding="utf-8")
    print(json.dumps({**_summary(log), "seconds": round(dt, 2)}, indent=2))
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    results = []
    for algo in ALGOS:
        for seed in args.seeds:
            cfg = RunConfig(algo=algo, seed=seed, iters=args.iters)
            t0 = time.time()
            log = run_algo(cfg)
            row = _summary(log)
            row["seed"] = seed
            row["seconds"] = round(time.time() - t0, 2)
            results.append(row)
            print(f"[{algo:9s} seed={seed}] reward={row['final_reward']:.3f} "
                  f"acc={row['final_acc']:.3f} {row['seconds']}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="grpore", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="run a single algorithm arm")
    tp.add_argument("--algo", choices=ALGOS, default="grpo")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--iters", type=int, default=PUBLISHED_ITERS)
    tp.add_argument("--jsonl", default=None)
    tp.set_defaults(func=cmd_train)

    ab = sub.add_parser("ablate", help="run every arm across seeds")
    ab.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ab.add_argument("--iters", type=int, default=PUBLISHED_ITERS)
    ab.add_argument("--out", default="results/ablation.json")
    ab.set_defaults(func=cmd_ablate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
