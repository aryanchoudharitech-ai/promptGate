"""Load versioned prompt templates from prompts/."""
import hashlib
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
PLACEHOLDER = "{log_line}"


def list_versions() -> list[str]:
    names = [p.stem for p in PROMPTS_DIR.glob("v*.txt") if re.fullmatch(r"v\d+", p.stem)]
    return sorted(names, key=lambda n: int(n[1:]))


def load_prompt(version: str) -> dict:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No prompt '{version}'. Available: {list_versions()}")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    if PLACEHOLDER not in text:
        raise ValueError(f"{path.name} is missing the {PLACEHOLDER} placeholder")
    return {
        "version": version,
        "template": text,
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def render_prompt(template: str, log_line: str) -> str:
    return template.replace(PLACEHOLDER, log_line)


if __name__ == "__main__":
    from src.provider import call_model, load_config

    line = "ERROR payment-svc: DB connection refused (host=db-1:5432), retrying in 5s"
    for v in list_versions():
        p = load_prompt(v)
        out = call_model(render_prompt(p["template"], line), load_config("tested"))
        print(v, p["hash"], "->", out["text"])