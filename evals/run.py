"""Run the 50-case evaluation and print a results table.

Usage:
    python evals/run.py                              # provider from env (default: offline rules baseline)
    python evals/run.py --provider ollama --model qwen3.5:4b
    python evals/run.py --provider groq --prompt-version v1   # before/after prompt comparison
    python evals/run.py --min-pass 0.8                # exit 1 if pass rate < 80% (CI)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from health_agent.agent import HealthAgent, Session  # noqa: E402
from health_agent.llm import make_client  # noqa: E402
from health_agent.prompts import PROMPT_VERSION  # noqa: E402
from health_agent.schemas import FIELDS  # noqa: E402

CASES = Path(__file__).with_name("cases.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")

GROUP_LABELS = {
    "complete_en": "Complete inputs (English)",
    "complete_hinglish": "Complete inputs (Hinglish)",
    "missing": "Missing fields -> follow-up",
    "what_if": "\"What if\" follow-ups",
    "out_of_range": "Out-of-range values caught",
    "refusal": "Off-topic / unsafe refused",
}


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 0.01


def score_case(case: dict, reply, session: Session) -> dict:
    exp = case["expect"]
    checks: dict[str, bool] = {}
    field_hits = field_total = invented = 0

    observed = (reply.inputs_used if reply.status == "result" and reply.inputs_used else reply.known_inputs).model_dump()
    if "fields" in exp:
        for f in FIELDS:
            want, got = exp["fields"].get(f), observed.get(f)
            field_total += 1
            field_hits += close(want, got)
            invented += want is None and got is not None
        checks["fields"] = field_hits == field_total

    checks["status"] = reply.status == exp["status"]

    ok_tools = [t for t in reply.tool_calls if t.ok]
    if exp.get("tool"):
        tool_ok = bool(ok_tools) and ok_tools[-1].name == exp["tool"]
        if tool_ok and exp.get("model"):
            tool_ok = ok_tools[-1].arguments.get("model") == exp["model"]
    else:
        tool_ok = not ok_tools
    checks["tool"] = tool_ok

    if "missing" in exp:
        checks["missing"] = sorted(reply.missing_fields) == sorted(exp["missing"])
    if "invalid" in exp:
        checks["invalid"] = set(exp["invalid"]) <= set(reply.invalid_fields)
    if exp.get("what_if"):
        checks["what_if"] = reply.what_if and all(close(getattr(session.inputs, k), v) for k, v in exp["saved_fields"].items())
    checks["disclaimer"] = reply.disclaimer.startswith("Educational demo")

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "field_hits": field_hits,
        "field_total": field_total,
        "invented": invented,
        "tool_correct": tool_ok,
        "rejected_tool_calls": sum(not t.ok for t in reply.tool_calls),
    }


async def run_case(agent: HealthAgent, case: dict, outage_retries: int = 3, outage_wait: float = 30.0) -> dict:
    """Run one conversation. If the provider is down (status=error), wait and
    re-run the whole case so outages are not scored as agent mistakes."""
    for attempt in range(outage_retries + 1):
        session = Session()
        latencies, tokens = [], 0
        reply = None
        for turn in case["turns"]:
            reply, m = await agent.handle(session, turn)
            latencies.append(m.latency_ms)
            tokens += m.total_tokens
            if reply.status == "error":
                break
        if reply.status != "error" or attempt == outage_retries:
            break
        print(f"  provider error on {case['id']}, retrying in {outage_wait:.0f}s", flush=True)
        await asyncio.sleep(outage_wait)
    result = score_case(case, reply, session)
    result.update(
        provider_error=reply.status == "error",
        id=case["id"],
        group=case["group"],
        latencies=latencies,
        tokens=tokens,
        status=reply.status,
        explanation=reply.explanation,
        tool_calls=[t.model_dump() for t in reply.tool_calls],
        observed=(reply.inputs_used or reply.known_inputs).model_dump(),
    )
    return result


def pct(a, b) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def percentile(values, p) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[k]


def group_field_accuracy(results: list[dict], group: str | None = None) -> float | None:
    rs = [r for r in results if group is None or r["group"] == group]
    fh, ft = sum(r["field_hits"] for r in rs), sum(r["field_total"] for r in rs)
    return fh / ft if ft else None


def summarize(results: list[dict], label: str) -> dict:
    outages = [r for r in results if r.get("provider_error")]
    results = [r for r in results if not r.get("provider_error")]
    if not results:
        print(f"\nAll {len(outages)} cases hit provider errors; no scores computed.")
        return {"label": label, "cases": 0, "provider_errors": len(outages), "pass_rate": 0.0}
    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    lines = [f"\n## Results: {label}\n", "| Test group | Cases | Passed | Field accuracy | Tool-call correct |", "|---|---|---|---|---|"]
    for g, label_g in GROUP_LABELS.items():
        rs = by_group.get(g, [])
        if not rs:
            continue
        fh, ft = sum(r["field_hits"] for r in rs), sum(r["field_total"] for r in rs)
        lines.append(f"| {label_g} | {len(rs)} | {pct(sum(r['passed'] for r in rs), len(rs))} | {pct(fh, ft)} | {pct(sum(r['tool_correct'] for r in rs), len(rs))} |")

    all_lat = [l for r in results for l in r["latencies"]]
    fh, ft = sum(r["field_hits"] for r in results), sum(r["field_total"] for r in results)
    summary = {
        "label": label,
        "cases": len(results),
        "pass_rate": sum(r["passed"] for r in results) / len(results),
        "field_accuracy": fh / ft if ft else None,
        "invented_values": sum(r["invented"] for r in results),
        "tool_call_accuracy": sum(r["tool_correct"] for r in results) / len(results),
        "rejected_tool_calls": sum(r["rejected_tool_calls"] for r in results),
        "latency_p50_ms": round(statistics.median(all_lat), 1) if all_lat else 0,
        "latency_p95_ms": round(percentile(all_lat, 95), 1),
        "tokens_per_conversation": round(sum(r["tokens"] for r in results) / len(results), 1),
        "provider_errors": len(outages),
        "hinglish_field_accuracy": group_field_accuracy(results, "complete_hinglish"),
        "english_field_accuracy": group_field_accuracy(results, "complete_en"),
        "refusal_accuracy": (lambda rs: sum(r["passed"] for r in rs) / len(rs) if rs else None)(by_group.get("refusal", [])),
    }

    def fmt(x):
        return f"{100 * x:.0f}%" if x is not None else "-"

    resume = [
        "\n## Resume numbers",
        f"- Field-extraction accuracy (overall): {fmt(summary['field_accuracy'])}",
        f"- Field-extraction accuracy (Hinglish): {fmt(summary['hinglish_field_accuracy'])}",
        f"- Tool-call accuracy: {fmt(summary['tool_call_accuracy'])}",
        f"- Safety refusals correct: {fmt(summary['refusal_accuracy'])}",
        f"- Invented values: {summary['invented_values']}",
        f"- Cases scored: {len(results)}/{len(results) + len(outages)}"
        + (f" ({len(outages)} left out after repeated provider outages)" if outages else ""),
    ]
    lines += [
        f"| **Overall** | {len(results)} | **{pct(sum(r['passed'] for r in results), len(results))}** | {pct(fh, ft)} | {pct(sum(r['tool_correct'] for r in results), len(results))} |",
        "",
        f"- Invented values (field filled that the user never gave): {summary['invented_values']}",
        f"- Tool calls rejected by the validation guard: {summary['rejected_tool_calls']}",
        f"- Latency per message: median {summary['latency_p50_ms']:.0f} ms, p95 {summary['latency_p95_ms']:.0f} ms",
        f"- Tokens per conversation: {summary['tokens_per_conversation']:.0f}",
    ]
    failed = [r for r in results if not r["passed"]]
    if failed:
        lines.append("\nFailed cases:")
        for r in failed:
            bad = [k for k, v in r["checks"].items() if not v]
            lines.append(f"- {r['id']} ({r['group']}): failed {', '.join(bad)}; status={r['status']}; observed={r['observed']}")
    if outages:
        lines.append("\nLeft out (provider outage): " + ", ".join(r["id"] for r in outages))
    print("\n".join(lines + resume))
    return summary


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-version", default=None, choices=["v1", "v2"])
    ap.add_argument("--group", default=None)
    ap.add_argument("--ids", default=None, help="comma-separated case ids")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-pass", type=float, default=0.0)
    args = ap.parse_args()

    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.group:
        cases = [c for c in cases if c["group"] == args.group]
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    client = make_client(args.provider, args.model)
    version = args.prompt_version or PROMPT_VERSION
    agent = HealthAgent(client, prompt_version=version)
    label = f"{client.name}, prompt {version}"
    print(f"Running {len(cases)} cases with {label}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()

    async def guarded(c):
        async with sem:
            r = await run_case(agent, c)
            print(f"  {'PASS' if r['passed'] else 'FAIL'} {r['id']}", flush=True)
            return r

    results = await asyncio.gather(*(guarded(c) for c in cases))
    summary = summarize(list(results), label)
    summary["wall_seconds"] = round(time.time() - start, 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{client.name.replace(':', '_').replace('/', '_')}_{version}.json"
    out.write_text(json.dumps({"summary": summary, "cases": list(results)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")
    return 0 if summary["pass_rate"] >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
