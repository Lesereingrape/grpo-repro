"""A bare ``grpore train`` must run the budget the README table was measured at."""

from __future__ import annotations

import json
from pathlib import Path

from grpore.cli import build_parser
from grpore.train import PUBLISHED_ITERS

ROOT = Path(__file__).resolve().parents[1]


def test_cli_iters_default_matches_the_committed_curves():
    records = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))
    measured = {r["iters"] for r in records}
    assert measured == {PUBLISHED_ITERS}
    for cmd in (["train"], ["ablate"]):
        args = build_parser().parse_args(cmd)
        assert args.iters == PUBLISHED_ITERS, cmd
