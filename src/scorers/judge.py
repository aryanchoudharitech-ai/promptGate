"""LLM-as-judge: scores what exact-match rules can't (is the classification reasonable?)."""
import json
import re
from pathlib import Path

from src.provider import call_model, load_config
from src.scorers.rules import check_schema, parse_output

JUDGE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "judge_v1.txt"


def load_judge_config() -> dict:
    judge, tested = load_config("judge"), load_config("tested")
    if judge["model"] == tested["model"]:
        raise ValueError("Judge and tested model are the same. Change one in config.yaml.")
    return judge


def _fill(template: str, **values: str) -> str:
    # One pass, so text inside a log line can never be re-substituted.
    return re.sub(r"\{(log_line|model_output)\}", lambda m: values[m.group(1)], template)


def _parse_judge_reply(text: str):
    match = re.search(r"\{.*\}", (text or "").strip(), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if data.get("verdict") in ("pass", "fail") else None


def judge_output(log_line: str, model_output: str, config: dict) -> dict:
    obj, err = parse_output(model_output)
    if err is None:
        err = check_schema(obj)
    if err:  # malformed output is an automatic fail, no API call needed
        return {"verdict": "fail", "reason": f"skipped judge: {err}", "skipped": True,
                "cost_usd": 0.0, "latency_ms": 0.0}

    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    prompt = _fill(template, log_line=log_line, model_output=model_output.strip())
    call = call_model(prompt, config)
    parsed = _parse_judge_reply(call["text"])
    base = {"skipped": False, "cost_usd": call["cost_usd"], "latency_ms": call["latency_ms"]}
    if parsed is None:
        return {**base, "verdict": "error", "reason": f"unparseable judge reply: {call['text'][:100]!r}"}
    return {**base, "verdict": parsed["verdict"], "reason": str(parsed.get("reason", ""))}