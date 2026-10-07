"""Compare the LLM judge to YOUR hand labels. Run only after labeling data/calibration.jsonl."""
import json
import logging
import sys
from pathlib import Path

from src.scorers.judge import judge_output, load_judge_config

CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.jsonl"


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    with open(CAL_PATH, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    unlabeled = [c["id"] for c in cases if c.get("human_verdict") not in ("pass", "fail")]
    if unlabeled:
        print(f'Label these first (human_verdict must be "pass" or "fail"): {unlabeled}')
        return 1

    cfg = load_judge_config()
    agree = too_lenient = too_strict = errors = 0
    cost = 0.0
    print(f"judge model: {cfg['model']}\n")
    print(f"{'id':5} {'human':6} {'judge':6} {'match':5}  reason")
    for c in cases:
        j = judge_output(c["log_line"], c["model_output"], cfg)
        cost += j["cost_usd"]
        match = j["verdict"] == c["human_verdict"]
        if match:
            agree += 1
        elif j["verdict"] == "pass":
            too_lenient += 1
        elif j["verdict"] == "fail":
            too_strict += 1
        else:
            errors += 1
        print(f"{c['id']:5} {c['human_verdict']:6} {j['verdict']:6} {'yes' if match else 'NO':5}  {j['reason'][:90]}")

    n = len(cases)
    print(f"\nagreement: {agree}/{n} ({agree / n:.0%})")
    print(f"judge too lenient (judge pass, you fail): {too_lenient}")
    print(f"judge too strict  (judge fail, you pass): {too_strict}")
    if errors:
        print(f"unparseable judge replies: {errors}")
    print(f"judge cost: ${cost:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())