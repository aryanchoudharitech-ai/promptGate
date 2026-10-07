"""Tests for the rule scorers. Pure functions: no network, runs in milliseconds."""
import pytest
from src.scorers.rules import RULE_KEYS, check_allowed, parse_output, score_output
from src.taxonomy import CATEGORIES, SEVERITIES

CASE = {"expected_category": "database", "expected_severity": "high"}


def score(text):
    return score_output(text, CASE)


def test_perfect_output_passes():
    r = score('{"category": "database", "severity": "high"}')
    assert r["passed"]
    assert r["errors"] == []


def test_wrong_severity_fails_only_the_severity_rule():
    r = score('{"category": "database", "severity": "low"}')
    assert not r["passed"]
    assert r["valid_json"] and r["schema_ok"] and r["allowed_values"] and r["category_correct"]
    assert not r["severity_correct"]
    assert r["errors"] == ["severity: got 'low', expected 'high'"]


def test_wrong_category_fails_only_the_category_rule():
    r = score('{"category": "network", "severity": "high"}')
    assert not r["passed"]
    assert r["severity_correct"] and not r["category_correct"]


@pytest.mark.parametrize("text", [
    "This is a database error.",
    '```json\n{"category": "database", "severity": "high"}\n```',
    '{"category": "database", "severity": "high"} and some trailing text',
    "",
    None,
])
def test_non_json_fails_every_rule(text):
    r = score(text)
    assert not r["passed"]
    assert not any(r[k] for k in RULE_KEYS)
    assert len(r["errors"]) == 1


@pytest.mark.parametrize("text", [
    '["database", "high"]',
    '{"category": "database"}',
    '{"category": "database", "severity": "high", "reason": "x"}',
    '{"category": 1, "severity": "high"}',
])
def test_bad_schema_fails_the_schema_rule(text):
    r = score(text)
    assert r["valid_json"]
    assert not r["schema_ok"]
    assert not r["passed"]


def test_labels_are_case_sensitive():
    r = score('{"category": "Database", "severity": "High"}')
    assert not r["allowed_values"]
    assert not r["passed"]


def test_invented_label_reports_one_error_not_two():
    # Regression test: the "got X, expected Y" message must not repeat when the label isn't allowed.
    r = score('{"category": "database", "severity": "error"}')
    assert r["errors"] == ["severity 'error' not allowed"]


def test_parse_output_returns_an_error_message_not_an_exception():
    obj, err = parse_output("nope")
    assert obj is None
    assert err.startswith("invalid JSON")


def test_check_allowed_accepts_every_taxonomy_value():
    for category in CATEGORIES:
        for severity in SEVERITIES:
            assert check_allowed({"category": category, "severity": severity}) is None