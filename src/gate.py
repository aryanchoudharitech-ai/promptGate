"""Regression gate: fail the build if a prompt scores worse than the committed baseline.

Exit codes: 0 = pass, 1 = prompt regressed, 2 = gate could not evaluate (setup or infra problem).
"""
import argparse
import hashlib
import json
import logging
import sys
import traceback
from pathlib import Path

from src.dataset import GOLDEN_PATH
from src.db import get_case_results, last_good_run
from src.provider import load_config
from src.run_eval import print_summary, run_eval, save_results

BASELINE_PATH = Path(__file__).resolve().parent.parent / "data" / "baseline.json"
DEFAULT_THRESHOLD = 5.0  # percentage POINTS, not relative percent
EXIT_OK, EXIT_REGRESSION, EXIT_CANNOT_EVALUATE = 0, 1, 2


class GateSetupError(Exception):
    """The gate can't give a fair verdict (missing or stale baseline, wrong model)."""


def dataset_hash() -> str:
    text = GOLDEN_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def export_baseline(path: Path = BASELINE_PATH) -> dict:
    run = last_good_run()
    if run is None:
        raise GateSetupError("No run is marked good. Run: python -m src.db good <run_id>")
    baseline = {
        "run_id": run["id"],
        "created_at": run["created_at"],
        "prompt_version": run["prompt_version"],
        "prompt_hash": run["prompt_hash"],
        "model": run["model"],
        "cases": run["cases"],
        "pass_rate": run["pass_rate"],
        "dataset_hash": dataset_hash(),
        "case_results": get_case_results(run["id"]),
    }
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    if not path.exists():
        raise GateSetupError(f"No baseline at {path}. Create one: python -m src.gate export-baseline")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline["dataset_hash"] != dataset_hash():
        raise GateSetupError(
            "golden.jsonl changed since the baseline was recorded. Re-run the eval, "
            "mark it good, and export a new baseline."
        )
    return baseline


def compare(baseline: dict, report: dict) -> dict:
    now = {r["id"]: r["score"]["passed"] for r in report["results"]}
    was = baseline["case_results"]
    return {
        "change_points": (report["summary"]["pass_rate"] - baseline["pass_rate"]) * 100,
        "regressed": sorted(c for c in now if was.get(c) and not now[c]),
        "fixed": sorted(c for c in now if now[c] and not was.get(c, False)),
    }


def describe(diff: dict) -> None:
    print(f"\nvs baseline: {diff['change_points']:+.1f} points")
    if diff["regressed"]:
        print(f"  regressed ({len(diff['regressed'])}): {', '.join(diff['regressed'])}")
    if diff["fixed"]:
        print(f"  fixed ({len(diff['fixed'])}): {', '.join(diff['fixed'])}")


def check(version: str, threshold: float, delay: float, baseline_path: Path) -> int:
    baseline = load_baseline(baseline_path)
    model = load_config("tested")["model"]
    if model != baseline["model"]:
        raise GateSetupError(
            f"tested model is {model} but the baseline was recorded on {baseline['model']}. Re-baseline."
        )
    print(f"baseline: run #{baseline['run_id']} ({baseline['prompt_version']}, "
          f"{baseline['pass_rate']:.1%} on {baseline['cases']} cases)")
    print(f"rule: fail if the pass rate drops more than {threshold:.1f} percentage points\n")

    for attempt in (1, 2):
        print(f"=== attempt {attempt} of 2 ===")
        report = run_eval(version, delay=delay)
        print_summary(report)
        print(f"saved: {save_results(report)}")
        diff = compare(baseline, report)
        describe(diff)
        if diff["change_points"] >= -threshold:
            print(f"\nGATE: PASS ({diff['change_points']:+.1f} points, limit -{threshold:.1f})")
            return EXIT_OK
        if attempt == 1:
            print("\nbelow the limit: re-running once to rule out noise\n")
    print(f"\nGATE: FAIL ({diff['change_points']:+.1f} points on both attempts, limit -{threshold:.1f})")
    return EXIT_REGRESSION


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Prompt regression gate.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="evaluate a prompt version against the baseline")
    c.add_argument("version", help="prompt version, e.g. v3")
    c.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    c.add_argument("--delay", type=float, default=3.5)
    c.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    sub.add_parser("export-baseline", help="write data/baseline.json from the last run marked good")
    args = ap.parse_args()

    try:
        if args.cmd == "export-baseline":
            b = export_baseline()
            print(f"wrote {BASELINE_PATH.name}: run #{b['run_id']}, {b['prompt_version']}, {b['pass_rate']:.1%}")
            return EXIT_OK
        return check(args.version, args.threshold, args.delay, args.baseline)
    except GateSetupError as e:
        print(f"GATE SETUP ERROR: {e}")
        return EXIT_CANNOT_EVALUATE


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - process boundary: a crash must not look like a regression (exit 1)
        traceback.print_exc()
        sys.exit(EXIT_CANNOT_EVALUATE)