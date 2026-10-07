"""Tests for the regression gate. run_eval is replaced by a fake, so there are no API calls."""
import json

import pytest
from src import gate

IDS = [f"c{i:02d}" for i in range(50)]


def make_report(passed: int) -> dict:
    """A fake eval report in which the first `passed` cases pass."""
    results = [{"id": cid, "score": {"passed": k < passed}} for k, cid in enumerate(IDS)]
    return {"results": results, "summary": {"pass_rate": passed / len(IDS)}}


def write_baseline(path, passed=38, model="test-model", ds_hash=None):
    baseline = {
        "run_id": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "prompt_version": "v2",
        "prompt_hash": "abc",
        "model": model,
        "cases": len(IDS),
        "pass_rate": passed / len(IDS),
        "dataset_hash": ds_hash or gate.dataset_hash(),
        "case_results": {cid: k < passed for k, cid in enumerate(IDS)},
    }
    path.write_text(json.dumps(baseline), encoding="utf-8")
    return path


@pytest.fixture
def run_gate(monkeypatch, tmp_path):
    """Run gate.check against a fake model. Returns (exit_code, number_of_eval_runs)."""
    monkeypatch.setattr(gate, "load_config", lambda profile: {"model": "test-model"})
    monkeypatch.setattr(gate, "print_summary", lambda report: None)

    def _run(baseline_passed, *attempt_passed, threshold=5.0):
        baseline_path = write_baseline(tmp_path / "baseline.json", passed=baseline_passed)
        reports = [make_report(p) for p in attempt_passed]
        calls = []

        def fake_run_eval(version, delay=0):
            calls.append(version)
            return reports[min(len(calls), len(reports)) - 1]

        monkeypatch.setattr(gate, "run_eval", fake_run_eval)
        code = gate.check("v9", threshold, 0, baseline_path)
        return code, len(calls)

    return _run


@pytest.fixture
def no_model_calls(monkeypatch):
    """Any call to run_eval fails the test: the gate must stop before spending API quota."""
    def boom(*args, **kwargs):
        raise AssertionError("run_eval must not be called when the gate can't judge fairly")

    monkeypatch.setattr(gate, "run_eval", boom)
    monkeypatch.setattr(gate, "load_config", lambda profile: {"model": "test-model"})


@pytest.mark.parametrize("candidate, expected_exit", [
    (38, 0),  # same as baseline (76%)
    (36, 0),  # -4 points: inside the limit
    (35, 1),  # -6 points: beyond the limit
    (20, 1),  # big regression
])
def test_exit_code_for_a_single_score(run_gate, candidate, expected_exit):
    code, _ = run_gate(38, candidate)
    assert code == expected_exit


def test_passing_run_evaluates_only_once(run_gate):
    assert run_gate(38, 38) == (gate.EXIT_OK, 1)


def test_improvement_passes(run_gate):
    assert run_gate(38, 45)[0] == gate.EXIT_OK


def test_noise_gets_a_second_chance(run_gate):
    assert run_gate(38, 30, 38) == (gate.EXIT_OK, 2)


def test_two_bad_attempts_fail_the_gate(run_gate):
    assert run_gate(38, 30, 31) == (gate.EXIT_REGRESSION, 2)


def test_missing_baseline_is_a_setup_error(no_model_calls, tmp_path):
    with pytest.raises(gate.GateSetupError, match="No baseline"):
        gate.check("v9", 5.0, 0, tmp_path / "missing.json")


def test_changed_dataset_is_a_setup_error(no_model_calls, tmp_path):
    path = write_baseline(tmp_path / "baseline.json", ds_hash="stale")
    with pytest.raises(gate.GateSetupError, match="golden.jsonl changed"):
        gate.check("v9", 5.0, 0, path)


def test_different_model_is_a_setup_error(no_model_calls, tmp_path):
    path = write_baseline(tmp_path / "baseline.json", model="some-other-model")
    with pytest.raises(gate.GateSetupError, match="baseline was recorded on"):
        gate.check("v9", 5.0, 0, path)


def test_compare_lists_regressed_and_fixed_cases():
    baseline = {"pass_rate": 0.5, "case_results": {"a": True, "b": True, "c": False, "d": False}}
    report = {
        "results": [
            {"id": "a", "score": {"passed": True}},
            {"id": "b", "score": {"passed": False}},
            {"id": "c", "score": {"passed": True}},
            {"id": "d", "score": {"passed": False}},
        ],
        "summary": {"pass_rate": 0.5},
    }
    diff = gate.compare(baseline, report)
    assert diff["regressed"] == ["b"]
    assert diff["fixed"] == ["c"]
    assert diff["change_points"] == 0


def test_dataset_hash_ignores_line_endings(tmp_path, monkeypatch):
    # Windows checkouts use CRLF, CI uses LF. The hash must not care.
    f = tmp_path / "golden.jsonl"
    monkeypatch.setattr(gate, "GOLDEN_PATH", f)
    f.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    lf_hash = gate.dataset_hash()
    f.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert gate.dataset_hash() == lf_hash