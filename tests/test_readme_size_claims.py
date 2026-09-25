"""Guard the README's two hand-written size claims.

Everything between the RESULTS markers is byte-pinned against the committed
artifact; "~15k-parameter" and "~560 lines" in the prose are the figures a reader
takes on trust, so they get measured against the code here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from grpore.policy import Policy
from grpore.train import RunConfig

ROOT = Path(__file__).resolve().parents[1]

TOLERANCE_PARAMS = 1000
TOLERANCE_LINES = 60

#: The four modules the README's "the whole comparison lives in" sentence links.
CORE_MODULES = ("adv.py", "train.py", "policy.py", "env.py")


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _core_lines() -> int:
    return sum(len((ROOT / "src" / "grpore" / name).read_text(encoding="utf-8")
                   .splitlines()) for name in CORE_MODULES)


def test_readme_parameter_claim_matches_the_policy():
    claim = int(re.search(r"~(\d+)k-parameter", _readme()).group(1))
    actual = sum(p.numel() for p in Policy(hidden=RunConfig().hidden).parameters())
    assert abs(actual / 1000 - claim) * 1000 <= TOLERANCE_PARAMS, (
        f"README says ~{claim}k parameters but the arms build a policy with "
        f"{actual:,} ({actual / 1000:.1f}k)")


def test_readme_line_claim_matches_the_core_modules():
    claim = int(re.search(r"lives in ~(\d+) lines", _readme()).group(1))
    actual = _core_lines()
    assert abs(actual - claim) <= TOLERANCE_LINES, (
        f"README says ~{claim} lines for the comparison itself; the four linked "
        f"modules are {actual} lines today")
