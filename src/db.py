"""SQLite storage for eval runs. One file, no server, nothing to install."""
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "results" / "promptgate.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    prompt_version  TEXT    NOT NULL,
    prompt_hash     TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    cases           INTEGER NOT NULL,
    passed          INTEGER NOT NULL,
    pass_rate       REAL    NOT NULL,
    errors          INTEGER NOT NULL,
    avg_latency_ms  REAL    NOT NULL,
    p95_latency_ms  REAL    NOT NULL,
    cost_usd        REAL    NOT NULL,
    judge_pass_rate REAL,
    good            INTEGER NOT NULL DEFAULT 0,
    UNIQUE (created_at, prompt_hash, model)
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id    INTEGER NOT NULL REFERENCES runs(id),
    case_id   TEXT    NOT NULL,
    case_type TEXT    NOT NULL,
    passed    INTEGER NOT NULL,
    output    TEXT,
    errors    TEXT,
    PRIMARY KEY (run_id, case_id)
);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_run(report: dict, db_path: Path = DB_PATH) -> int:
    s = report["summary"]
    judge_rate = s["judge_passed"] / s["judge_cases"] if s["judge_cases"] else None
    with closing(connect(db_path)) as conn, conn:  # one transaction: both tables or neither
            cur = conn.execute(
                """INSERT INTO runs (created_at, prompt_version, prompt_hash, model, cases, passed,
                       pass_rate, errors, avg_latency_ms, p95_latency_ms, cost_usd, judge_pass_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (report["created_at"], report["version"], report["prompt_hash"], report["model"],
                 s["cases"], s["passed"], s["pass_rate"], s["errors"], s["avg_latency_ms"],
                 s["p95_latency_ms"], s["total_cost_usd"], judge_rate),
            )
            run_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO case_results (run_id, case_id, case_type, passed, output, errors) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, r["id"], r["type"], int(r["score"]["passed"]), r["output"],
                  r["error"] or "; ".join(r["score"]["errors"])) for r in report["results"]],
            )
    return run_id


def list_runs(db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    with closing(connect(db_path)) as conn:
        return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()


def last_good_run(db_path: Path = DB_PATH):
    with closing(connect(db_path)) as conn:
        return conn.execute("SELECT * FROM runs WHERE good = 1 ORDER BY id DESC LIMIT 1").fetchone()


def get_case_results(run_id: int, db_path: Path = DB_PATH) -> dict[str, bool]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT case_id, passed FROM case_results WHERE run_id = ?", (run_id,)
        ).fetchall()
    return {r["case_id"]: bool(r["passed"]) for r in rows}


def mark_good(run_id: int, db_path: Path = DB_PATH) -> None:
    with closing(connect(db_path)) as conn, conn:
            cur = conn.execute("UPDATE runs SET good = 1 WHERE id = ?", (run_id,))
            if cur.rowcount == 0:
                raise ValueError(f"No run #{run_id}")


def print_history() -> None:
    runs = list_runs()
    if not runs:
        print("no runs stored yet")
        return
    print(f"{'id':>3}  {'created (UTC)':19}  {'prompt':6}  {'hash':12}  {'model':20}  "
          f"{'cases':>5}  {'pass':>6}  {'err':>3}  {'avg ms':>6}  {'cost $':>8}  good")
    for r in runs:
        print(f"{r['id']:>3}  {r['created_at'][:19]:19}  {r['prompt_version']:6}  {r['prompt_hash']:12}  "
              f"{r['model']:20}  {r['cases']:>5}  {r['pass_rate']:>6.1%}  {r['errors']:>3}  "
              f"{r['avg_latency_ms']:>6.0f}  {r['cost_usd']:>8.5f}  {'*' if r['good'] else ''}")
    with closing(connect()) as conn:
        n = conn.execute("SELECT COUNT(*) FROM case_results").fetchone()[0]
    print(f"\n{len(runs)} runs, {n} case rows")


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(errors="replace")
    if not argv:
        print_history()
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "import" and len(args) == 1:
        report = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        try:
            run_id = save_run(report)
        except sqlite3.IntegrityError:
            print("already stored: that report is in the database")
            return 1
        print(f"stored run #{run_id} ({report['version']}, {report['summary']['pass_rate']:.1%})")
        return 0
    if cmd == "good" and len(args) == 1:
        mark_good(int(args[0]))
        print(f"run #{args[0]} marked good")
        return 0
    print("usage: python -m src.db | import <report.json> | good <run_id>")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))