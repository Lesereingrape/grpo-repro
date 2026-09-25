"""The committed artifact must agree with itself, before anyone agrees with it.

The README block is generated from ``results/ablation.json``, so a wrong number in the
JSON is a wrong number in the README with a clean CI. These checks recompute the
summary fields from the raw curves the runs carry, and assert the artifact records the
environment it claims to be reproducible under.
"""

from __future__ import annotations

import json
from pathlib import Path

from grpore.study import ALGO_ORDER, PUBLISHED_SEEDS, environment, published_config

ROOT = Path(__file__).resolve().parents[1]

DATA = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))
EVAL_EVERY = DATA["config"]["eval_every"]


def test_artifact_has_the_published_shape():
    assert set(DATA) == {"environment", "config", "runtime_sec", "runs"}
    assert list(ALGO_ORDER) == DATA["config"]["algos"]
    assert DATA["config"]["seeds"] == list(PUBLISHED_SEEDS)
    seen = {(r["algo"], r["seed"]) for r in DATA["runs"]}
    assert seen == {(a, s) for a in ALGO_ORDER for s in PUBLISHED_SEEDS}, (
        "the artifact must hold one run per (algo, seed) or the means are not the "
        "same average the README claims")


def test_summary_fields_recompute_from_the_curves():
    for run in DATA["runs"]:
        curve, acc = run["reward_curve"], run["acc_curve"]
        assert len(curve) == len(acc)
        # the printed curves are rounded to 4 dp; the scalars are the raw values
        assert abs(run["start_reward"] - curve[0]) <= 5e-5, run["algo"]
        assert abs(run["final_reward"] - curve[-1]) <= 5e-5, run["algo"]
        assert abs(run["final_acc"] - acc[-1]) <= 5e-5, run["algo"]
        assert run["iters"] == DATA["config"]["iters"]
        # one eval at iteration 0 plus one per eval_every, so a curve cannot be
        # truncated without this failing
        assert len(curve) == DATA["config"]["iters"] // EVAL_EVERY + 1, run["algo"]


def test_curves_start_near_zero_and_end_above_start():
    """A run whose reward did not move is not evidence about the update rule."""
    for run in DATA["runs"]:
        assert run["start_reward"] < 0.05, (
            f"{run['algo']} seed {run['seed']} started at "
            f"{run['start_reward']:.3f}; the untrained policy is supposed to be near "
            "the floor, and if it is not, 'the arm improved' is not a finding")
        assert run["final_reward"] > run["start_reward"], (
            f"{run['algo']} seed {run['seed']} ended where it started")


def test_environment_block_is_recorded_and_readable():
    """Shape, not values: CI installs its own torch, and a bump must not go red.

    The point of the block is that a reader can see *which* machine produced the
    numbers, so the test only pins that the fields exist and are plausible. The
    README sentence that quotes them is byte-checked by the drift test.
    """
    env = DATA["environment"]
    assert set(env) == set(environment())
    assert env["threads"] >= 1
    assert env["device"] == "cpu"
    assert env["python"] and env["torch"] and env["platform"]


def test_config_is_the_code_own_defaults():
    assert DATA["config"] == published_config()
