from __future__ import annotations
import json
import sqlite3
from pathlib import Path

from mcpeval.runner import CaseResult, RunResult

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    eval_suite TEXT,
    model TEXT,
    total_cases INTEGER,
    passed INTEGER,
    failed INTEGER,
    overall_score REAL
)
"""

_CREATE_CASE_RESULTS = """
CREATE TABLE IF NOT EXISTS case_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    case_id TEXT,
    passed BOOLEAN,
    tool_calls_made TEXT,
    tool_calls_expected TEXT,
    graph_match_score REAL,
    llm_judge_score REAL,
    steps_taken INTEGER,
    terminated_cleanly BOOLEAN,
    raw_output TEXT
)
"""


class ResultStore:
    def __init__(self, db_path: str | Path = "mcpeval.db") -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_RUNS)
            conn.execute(_CREATE_CASE_RESULTS)
            conn.commit()

    def save_run(self, result: RunResult) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (eval_suite, model, total_cases, passed, failed, overall_score) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (result.eval_suite, result.model, result.total_cases,
                 result.passed, result.failed, result.overall_score),
            )
            conn.commit()
            return cursor.lastrowid

    def save_case_result(self, run_id: int, cr: CaseResult) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO case_results "
                "(run_id, case_id, passed, tool_calls_made, tool_calls_expected, "
                "graph_match_score, llm_judge_score, steps_taken, terminated_cleanly, raw_output) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    cr.case_id,
                    int(cr.passed),
                    json.dumps(cr.tool_calls_made),
                    json.dumps(cr.tool_calls_expected),
                    cr.graph_match_score,
                    cr.llm_judge_score,
                    cr.steps_taken,
                    int(cr.terminated_cleanly),
                    cr.raw_output,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_run(self, run_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def get_case_results(self, run_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM case_results WHERE run_id = ?", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_at DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
