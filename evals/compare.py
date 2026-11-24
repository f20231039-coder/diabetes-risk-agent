"""Before/after comparison of two eval runs (e.g. prompt v1 vs v2).

Usage:
    python evals/compare.py evals/results/gemini_gemini-3.5-flash_v1.json evals/results/gemini_gemini-3.5-flash_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.run import GROUP_LABELS, group_field_accuracy  # noqa: E402


def load(path: str) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if not c.get("provider_error")]
    return data["summary"], cases


def pct(x) -> str:
    return f"{100 * x:.0f}%" if x is not None else "-"


def pass_rate(cases, group=None):
    rs = [c for c in cases if group is None or c["group"] == group]
    return sum(c["passed"] for c in rs) / len(rs) if rs else None


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    (sb, before), (sa, after) = load(sys.argv[1]), load(sys.argv[2])
    print(f"Before: {sb['label']}\nAfter:  {sa['label']}\n")
    print("| Metric | Before | After |\n|---|---|---|")
    print(f"| Field accuracy (overall) | {pct(group_field_accuracy(before))} | {pct(group_field_accuracy(after))} |")
    for g in ("complete_en", "complete_hinglish", "missing", "what_if"):
        print(f"| Field accuracy: {GROUP_LABELS[g]} | {pct(group_field_accuracy(before, g))} | {pct(group_field_accuracy(after, g))} |")
    print(f"| Tool-call accuracy | {pct(sb['tool_call_accuracy'])} | {pct(sa['tool_call_accuracy'])} |")
    for g in GROUP_LABELS:
        print(f"| Pass rate: {GROUP_LABELS[g]} | {pct(pass_rate(before, g))} | {pct(pass_rate(after, g))} |")
    print(f"| **Pass rate (overall)** | {pct(pass_rate(before))} | {pct(pass_rate(after))} |")
    print(f"| Invented values | {sb['invented_values']} | {sa['invented_values']} |")


if __name__ == "__main__":
    main()
