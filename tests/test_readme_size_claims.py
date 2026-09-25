"""Guard the README's hand-written claims that the renderer does not write.

Everything between the RESULTS markers is byte-pinned against the committed
artifact; "~15k-parameter", "~560 lines", the wall-clock the opening paragraph
promises and the training budget in the honesty section are the figures a reader
takes on trust, so they get measured against the code and the artifact here.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from grpore.policy import Policy
from grpore.train import RunConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))

TOLERANCE_PARAMS = 1000
TOLERANCE_LINES = 60

#: The four modules the README's "the whole comparison lives in" sentence links.
CORE_MODULES = ("adv.py", "train.py", "policy.py", "env.py")


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _prose() -> str:
    """The README outside the generated block, which has its own byte-pinning."""
    readme = _readme()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    return readme.split(start, 1)[0] + readme.split(end, 1)[1]


def _core_lines() -> int:
    return sum(len((ROOT / "src" / "grpore" / name).read_text(encoding="utf-8")
                   .splitlines()) for name in CORE_MODULES)


def _slowest_arm_minutes() -> float:
    """Mean wall-clock of the slowest arm, the only one the blurb could mean."""
    per_algo: dict[str, list[float]] = {}
    for run in DATA["runs"]:
        per_algo.setdefault(run["algo"], []).append(run["seconds"])
    return max(statistics.fmean(s) for s in per_algo.values()) / 60.0


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


def test_the_blurb_wall_clock_is_the_measured_wall_clock():
    """The opening paragraph used to promise "CPU seconds" for a two-minute arm."""
    numerals = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    m = re.search(r"in \*\*about (\d+|one|two|three|four|five|six) CPU minutes?\*\*",
                  _prose())
    assert m, "the README no longer states how long the slowest arm takes"
    claimed = float(numerals.get(m.group(1), m.group(1)))
    measured = _slowest_arm_minutes()
    assert 0.5 * claimed <= measured <= 2.0 * claimed, (
        f"the blurb promises ~{claimed:g} CPU minutes; the committed runs of the "
        f"slowest arm averaged {measured:.1f} minutes")


def test_the_honesty_section_names_the_budget_that_ran():
    m = re.search(r"the budget is (\d+)\s*\n?iterations", _prose())
    assert m, "the scope section no longer states the training budget"
    assert int(m.group(1)) == DATA["config"]["iters"], (
        f"prose says {m.group(1)} iterations, the artifact ran "
        f"{DATA['config']['iters']}")
