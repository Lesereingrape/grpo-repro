"""A bare ``grpore train`` must run the budget the README table was measured at."""

from __future__ import annotations

import json
from pathlib import Path

from grpore.cli import build_parser
from grpore.study import PUBLISHED_SEEDS
from grpore.train import PUBLISHED_ITERS

ROOT = Path(__file__).resolve().parents[1]


def test_cli_iters_default_matches_the_committed_curves():
    artifact = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))
    records = artifact["runs"]
    measured = {r["iters"] for r in records}
    assert measured == {PUBLISHED_ITERS}
    assert artifact["config"]["iters"] == PUBLISHED_ITERS
    for cmd in (["train"], ["ablate"]):
        args = build_parser().parse_args(cmd)
        assert args.iters == PUBLISHED_ITERS, cmd


def test_ablate_default_writes_the_committed_artifact_path():
    """The command in the README must land on the file the README block reads."""
    args = build_parser().parse_args(["ablate"])
    assert args.out == "results/ablation.json"
    assert tuple(args.seeds) == PUBLISHED_SEEDS
