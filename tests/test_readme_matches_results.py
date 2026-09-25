"""The README results block must be exactly what make_report renders from the JSON.

Guards the repo's central claim: every number in the table is machine-generated from
``results/ablation.json`` and nothing is hand-copied into prose.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_renderer():
    path = ROOT / "experiments" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(text: str) -> str:
    m = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", text, re.DOTALL)
    assert m, "README is missing the RESULTS:START/END block"
    return m.group(1).strip()


def test_readme_matches_committed_results():
    data = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))
    rendered = _load_renderer().build(data).strip()
    readme_block = _block((ROOT / "README.md").read_text(encoding="utf-8"))
    assert readme_block == rendered, (
        "README results drift: run `python experiments/make_report.py --write` to "
        "splice the rendered block back into README.md."
    )


def test_renderer_quotes_no_number_that_is_not_in_the_artifact():
    """The block may only print figures the artifact carries.

    ``--write`` makes regeneration easy enough that a stale hand-edit in the
    renderer is the remaining failure mode, so the table cells are re-derived here
    from the raw runs and compared with the rendered strings.
    """
    import statistics

    data = json.loads((ROOT / "results" / "ablation.json").read_text(encoding="utf-8"))
    block = _block((ROOT / "README.md").read_text(encoding="utf-8"))
    for algo, runs in _groups(data).items():
        mean = statistics.mean(r["final_reward"] for r in runs)
        sd = statistics.stdev([r["final_reward"] for r in runs])
        cell = f"{mean:.3f} ± {sd:.3f}"
        rows = [line for line in block.splitlines() if line.startswith("|")]
        assert any(cell in line and _label_of(algo) in line for line in rows), (
            f"{algo}: {cell} is not the artifact's mean ± sample sd, so the README "
            "table is not being regenerated from it"
        )


def _groups(data: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for run in data["runs"]:
        out.setdefault(run["algo"], []).append(run)
    return out


def _label_of(algo: str) -> str:
    return {"grpo": "GRPO", "reinforce": "REINFORCE", "dpo": "DPO"}[algo]
