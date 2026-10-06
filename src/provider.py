"""Single entry point for every LLM call in PromptGate."""
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import groq
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
MAX_RETRIES = 5


def load_config(profile: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        profiles = yaml.safe_load(f)
    if profile not in profiles:
        raise KeyError(f"Profile '{profile}' not in config.yaml. Found: {list(profiles)}")
    return profiles[profile]


@lru_cache(maxsize=1)
def _groq_client() -> groq.Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing. Check your .env file.")
    return groq.Groq(api_key=key, max_retries=0)  # we handle retries ourselves


def _call_groq(prompt: str, cfg: dict):
    extra = {}
    if "reasoning_effort" in cfg:
        extra["reasoning_effort"] = cfg["reasoning_effort"]
    resp = _groq_client().chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.get("temperature", 0),
        max_tokens=cfg.get("max_tokens", 200),
        **extra,
    )
    choice = resp.choices[0]
    if not choice.message.content and choice.finish_reason == "length":
        raise RuntimeError("Model hit max_tokens before answering. Raise max_tokens in config.yaml.")
    return (
        choice.message.content or "",
        resp.usage.prompt_tokens,
        resp.usage.completion_tokens,
    )


def _call_ollama(prompt: str, cfg: dict):
    try:
        r = requests.post(
            f"{cfg.get('base_url', 'http://localhost:11434')}/api/chat",
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": cfg.get("temperature", 0),
                    "num_predict": cfg.get("max_tokens", 200),
                },
            },
            timeout=120,
        )
        r.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError("Can't reach Ollama. Is it running? Try: ollama serve")
    data = r.json()
    return (
        data["message"]["content"],
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


PROVIDERS = {"groq": _call_groq, "ollama": _call_ollama}


def call_model(prompt: str, config: dict) -> dict:
    provider = config["provider"]
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Use: {list(PROVIDERS)}")
    fn = PROVIDERS[provider]

    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            text, tok_in, tok_out = fn(prompt, config)
            break
        except groq.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 * 2**attempt  # 2, 4, 8, 16, 32 seconds
            log.warning("rate limited, retry %d/%d in %ds", attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    latency_ms = (time.perf_counter() - start) * 1000

    cost = (
        tok_in * config.get("price_per_1m_input", 0)
        + tok_out * config.get("price_per_1m_output", 0)
    ) / 1_000_000

    result = {
        "text": text,
        "provider": provider,
        "model": config["model"],
        "latency_ms": round(latency_ms, 1),
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "cost_usd": cost,
        "retries": attempt,
    }
    log.info(
        "model=%s latency=%.0fms tokens_in=%d tokens_out=%d cost=$%.6f retries=%d",
        result["model"], latency_ms, tok_in, tok_out, cost, attempt,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    profile = sys.argv[1] if len(sys.argv) > 1 else "tested"
    prompt = 'Reply with JSON only, keys "category" and "severity". Log line: ERROR db connection timeout after 30s'
    out = call_model(prompt, load_config(profile))
    print(out["text"])