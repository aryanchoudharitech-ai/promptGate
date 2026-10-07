"""Deterministic rule scorers. No LLM involved: same input, same verdict, free."""
import json

from src.taxonomy import CATEGORIES, SEVERITIES

EXPECTED_KEYS = {"category", "severity"}
RULE_KEYS = ("valid_json", "schema_ok", "allowed_values", "category_correct", "severity_correct")


def parse_output(text: str):
    """Strict parse. Code fences, prose, or trailing text all count as failures."""
    try:
        return json.loads((text or "").strip()), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


def check_schema(obj) -> str | None:
    if not isinstance(obj, dict):
        return f"expected a JSON object, got {type(obj).__name__}"
    if set(obj) != EXPECTED_KEYS:
        return f"keys must be exactly {sorted(EXPECTED_KEYS)}, got {sorted(obj)}"
    if not all(isinstance(obj[k], str) for k in EXPECTED_KEYS):
        return "category and severity must both be strings"
    return None


def check_allowed(obj) -> str | None:
    problems = []
    if obj["category"] not in CATEGORIES:
        problems.append(f"category '{obj['category']}' not allowed")
    if obj["severity"] not in SEVERITIES:
        problems.append(f"severity '{obj['severity']}' not allowed")
    return "; ".join(problems) or None


def score_output(text: str, case: dict) -> dict:
    result = {k: False for k in RULE_KEYS}
    result["passed"] = False
    result["errors"] = []

    obj, err = parse_output(text)
    if err:
        result["errors"].append(err)
        return result
    result["valid_json"] = True

    err = check_schema(obj)
    if err:
        result["errors"].append(err)
        return result
    result["schema_ok"] = True

    result["category_correct"] = obj["category"] == case["expected_category"]
    result["severity_correct"] = obj["severity"] == case["expected_severity"]

    err = check_allowed(obj)
    if err:
        result["errors"].append(err)
    else:
        result["allowed_values"] = True
        if not result["category_correct"]:
            result["errors"].append(f"category: got '{obj['category']}', expected '{case['expected_category']}'")
        if not result["severity_correct"]:
            result["errors"].append(f"severity: got '{obj['severity']}', expected '{case['expected_severity']}'")

    result["passed"] = all(result[k] for k in RULE_KEYS)
    return result


if __name__ == "__main__":
    case = {"expected_category": "database", "expected_severity": "high"}
    samples = {
        "perfect": '{"category": "database", "severity": "high"}',
        "wrong severity": '{"category": "database", "severity": "low"}',
        "capitalized": '{"category": "Database", "severity": "High"}',
        "extra key": '{"category": "database", "severity": "high", "reason": "x"}',
        "code fence": '```json\n{"category": "database", "severity": "high"}\n```',
        "prose": "This is a database error.",
        "made-up label": '{"category": "database", "severity": "error"}',
    }
    for name, text in samples.items():
        r = score_output(text, case)
        print(f"{name:15} passed={r['passed']!s:5} {r['errors']}")