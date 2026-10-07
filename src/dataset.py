"""Load and validate the golden dataset."""
import json
import sys
from collections import Counter
from pathlib import Path

from src.prompts import PROMPTS_DIR
from src.taxonomy import CATEGORIES, SEVERITIES

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden.jsonl"
REQUIRED = {"id", "type", "log_line", "expected_category", "expected_severity"}
TYPES = {"normal", "edge", "adversarial"}


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} line {n}: invalid JSON ({e})")
    return cases


def few_shot_lines() -> set[str]:
    """Every 'Log: ...' example baked into a prompt file."""
    found = set()
    for p in PROMPTS_DIR.glob("v*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("Log: "):
                found.add(line[len("Log: "):].strip())
    return found


def validate(cases: list[dict]) -> list[str]:
    if not cases:
        return ["dataset is empty (0 cases)"]
    errors = []
    seen_ids, seen_lines = set(), set()
    leaked = few_shot_lines()
    for c in cases:
        cid = c.get("id", "?")
        missing = REQUIRED - c.keys()
        if missing:
            errors.append(f"{cid}: missing keys {sorted(missing)}")
            continue
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate id")
        seen_ids.add(cid)
        if c["type"] not in TYPES:
            errors.append(f"{cid}: type '{c['type']}' not in {sorted(TYPES)}")
        if c["expected_category"] not in CATEGORIES:
            errors.append(f"{cid}: category '{c['expected_category']}' not in taxonomy")
        if c["expected_severity"] not in SEVERITIES:
            errors.append(f"{cid}: severity '{c['expected_severity']}' not in taxonomy")
        line = c["log_line"].strip()
        if not line:
            errors.append(f"{cid}: empty log_line")
        if line in seen_lines:
            errors.append(f"{cid}: duplicate log_line")
        seen_lines.add(line)
        if line in leaked:
            errors.append(f"{cid}: log_line is a few-shot example in a prompt (leakage)")
    return errors


if __name__ == "__main__":
    cases = load_golden()
    errors = validate(cases)
    print(f"{len(cases)} cases")
    print("by type:", dict(Counter(c["type"] for c in cases)))
    print("by category:", dict(Counter(c["expected_category"] for c in cases)))
    print("by severity:", dict(Counter(c["expected_severity"] for c in cases)))
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("golden.jsonl OK")