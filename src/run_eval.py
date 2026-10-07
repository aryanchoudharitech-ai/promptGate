"""Run the golden dataset through one prompt version and print a summary."""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import groq
import requests

from src.dataset import load_golden, validate
from src.db import save_run
from src.prompts import load_prompt, render_prompt
from src.provider import call_model, load_config
from src.scorers.judge import judge_output, load_judge_config
from src.scorers.rules import RULE_KEYS, score_output

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_CONSECUTIVE_ERRORS = 5
# Failures we expect from a model call. Anything else (TypeError, KeyError, ...) is a bug: let it crash.
EXPECTED_ERRORS = (groq.APIError, requests.RequestException, RuntimeError)


def pick_cases(cases: list[dict], limit: int | None) -> list[dict]:
    """A limited sample is spread evenly so it still covers normal, edge and adversarial."""
    if not limit or limit >= len(cases):
        return cases
    step = len(cases) / limit
    return [cases[int(i * step)] for i in range(limit)]


def summarize(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(r["score"]["passed"] for r in results)
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["type"]][0] += r["score"]["passed"]
        by_type[r["type"]][1] += 1
    latencies = sorted(r["latency_ms"] for r in results if not r["error"])
    judged = [r["judge"] for r in results if "judge" in r]
    tested_cost = sum(r["cost_usd"] for r in results)
    judge_cost = sum(j["cost_usd"] for j in judged)
    return {
        "cases": n,
        "passed": passed,
        "pass_rate": passed / n,
        "rules": {k: sum(r["score"][k] for r in results) for k in RULE_KEYS},
        "by_type": {t: {"passed": p, "total": c} for t, (p, c) in by_type.items()},
        "errors": sum(1 for r in results if r["error"]),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
        "tested_cost_usd": tested_cost,
        "judge_cost_usd": judge_cost,
        "total_cost_usd": tested_cost + judge_cost,
        "judge_passed": sum(j["verdict"] == "pass" for j in judged),
        "judge_cases": len(judged),
    }


def run_eval(version: str, profile: str = "tested", use_judge: bool = False,
             limit: int | None = None, delay: float = 3.5) -> dict:
    all_cases = load_golden()
    problems = validate(all_cases)
    if problems:
        raise ValueError(f"golden.jsonl has problems, fix them first: {problems[:3]}")
    cases = pick_cases(all_cases, limit)

    prompt = load_prompt(version)
    model_cfg = load_config(profile)
    judge_cfg = load_judge_config() if use_judge else None

    results, streak = [], 0
    for i, case in enumerate(cases, start=1):
        if i > 1:
            time.sleep(delay)  # stay under the free-tier per-minute limits
        row = {"id": case["id"], "type": case["type"], "log_line": case["log_line"], "error": None}
        try:
            call = call_model(render_prompt(prompt["template"], case["log_line"]), model_cfg)
            streak = 0
        except EXPECTED_ERRORS as e:
            streak += 1
            row.update(output="", error=f"{type(e).__name__}: {e}", latency_ms=0.0, cost_usd=0.0,
                       input_tokens=0, output_tokens=0, score=score_output("", case))
            results.append(row)
            print(f"[{i:>2}/{len(cases)}] {case['id']:4} ERROR {row['error'][:80]}")
            if streak >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(f"{streak} errors in a row, stopping. Last: {row['error']}")
            continue

        row.update(output=call["text"], latency_ms=call["latency_ms"], cost_usd=call["cost_usd"],
                   input_tokens=call["input_tokens"], output_tokens=call["output_tokens"],
                   score=score_output(call["text"], case))
        if judge_cfg:
            try:
                row["judge"] = judge_output(case["log_line"], call["text"], judge_cfg)
            except EXPECTED_ERRORS as e:
                row["judge"] = {"verdict": "error", "reason": f"{type(e).__name__}: {e}",
                                "skipped": False, "cost_usd": 0.0, "latency_ms": 0.0}
        results.append(row)
        mark = "pass" if row["score"]["passed"] else "FAIL"
        print(f"[{i:>2}/{len(cases)}] {case['id']:4} {mark} {call['latency_ms']:.0f}ms")

    return {
        "version": version,
        "prompt_hash": prompt["hash"],
        "profile": profile,
        "model": model_cfg["model"],
        "judge_model": judge_cfg["model"] if judge_cfg else None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summarize(results),
        "results": results,
    }


def print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"\nPromptGate eval | prompt {report['version']} ({report['prompt_hash']}) | model {report['model']} | {s['cases']} cases")
    print(f"\npass rate: {s['passed']}/{s['cases']} ({s['pass_rate']:.1%})")
    for k, v in s["rules"].items():
        print(f"  {k:17} {v}/{s['cases']}  {v / s['cases']:.1%}")
    types = " | ".join(f"{t} {d['passed']}/{d['total']}" for t, d in s["by_type"].items())
    print(f"\nby type: {types}")
    if s["judge_cases"]:
        print(f"judge (advisory): {s['judge_passed']}/{s['judge_cases']} pass")
    print(f"latency: avg {s['avg_latency_ms']:.0f}ms, p95 {s['p95_latency_ms']:.0f}ms | errors: {s['errors']}")
    print(f"cost: ${s['total_cost_usd']:.5f} (tested ${s['tested_cost_usd']:.5f}, judge ${s['judge_cost_usd']:.5f})")

    failures = [r for r in report["results"] if not r["score"]["passed"]]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures[:15]:
            why = r["error"] or "; ".join(r["score"]["errors"]) or "unknown"
            print(f"  {r['id']} [{r['type']}] {why[:110]}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more (see the saved results file)")


def save_results(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{report['version']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Run the golden dataset through a prompt version.")
    ap.add_argument("version", help="prompt version, e.g. v1")
    ap.add_argument("--profile", default="tested", help="model profile from config.yaml")
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge (advisory)")
    ap.add_argument("--limit", type=int, help="run an evenly spread sample of N cases")
    ap.add_argument("--delay", type=float, default=3.5, help="seconds between model calls")
    args = ap.parse_args()

    report = run_eval(args.version, args.profile, args.judge, args.limit, args.delay)
    print_summary(report)
    print(f"\nsaved: {save_results(report)}")
    if args.limit:
        print("limit run: not stored in the database (partial samples aren't comparable)")
    else:
        print(f"stored in database as run #{save_run(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())