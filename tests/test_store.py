import json
import sqlite3
import pytest
from pathlib import Path
from mcpeval.store import ResultStore
from mcpeval.runner import RunResult, CaseResult


def _make_run_result(score: float = 0.9, cases: list[CaseResult] | None = None) -> RunResult:
    if cases is None:
        cases = [CaseResult(
            case_id="c001",
            passed=True,
            tool_calls_made=[{"tool_name": "get_logs", "arguments": {}}],
            tool_calls_expected=[{"tool": "get_logs"}],
            graph_match_score=score,
            llm_judge_score=None,
            steps_taken=2,
            terminated_cleanly=True,
            raw_output="Done.",
        )]
    return RunResult(
        run_id=None,
        eval_suite="My Suite",
        model="claude-3-5-haiku-20241022",
        total_cases=len(cases),
        passed=sum(1 for c in cases if c.passed),
        failed=sum(1 for c in cases if not c.passed),
        overall_score=score,
        case_results=cases,
    )


def test_initialize_creates_tables(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    conn = sqlite3.connect(tmp_db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "runs" in tables
    assert "case_results" in tables


def test_initialize_idempotent(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    store.initialize()  # should not raise


def test_save_run_returns_id(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    run_id = store.save_run(_make_run_result())
    assert isinstance(run_id, int)
    assert run_id >= 1


def test_save_run_persists_fields(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    run_id = store.save_run(_make_run_result(score=0.85))
    row = store.get_run(run_id)
    assert row is not None
    assert row["eval_suite"] == "My Suite"
    assert row["model"] == "claude-3-5-haiku-20241022"
    assert abs(row["overall_score"] - 0.85) < 0.001
    assert row["total_cases"] == 1
    assert row["passed"] == 1
    assert row["failed"] == 0


def test_save_case_result(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    result = _make_run_result()
    run_id = store.save_run(result)
    row_id = store.save_case_result(run_id, result.case_results[0])
    assert isinstance(row_id, int)
    rows = store.get_case_results(run_id)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c001"
    assert rows[0]["passed"] == 1


def test_tool_calls_made_is_valid_json(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    result = _make_run_result()
    run_id = store.save_run(result)
    store.save_case_result(run_id, result.case_results[0])
    rows = store.get_case_results(run_id)
    calls = json.loads(rows[0]["tool_calls_made"])
    assert isinstance(calls, list)
    assert calls[0]["tool_name"] == "get_logs"


def test_list_runs_ordered_desc(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    store.save_run(_make_run_result(score=0.5))
    store.save_run(_make_run_result(score=0.8))
    runs = store.list_runs()
    assert len(runs) == 2
    assert runs[0]["overall_score"] == pytest.approx(0.8)


def test_get_run_returns_none_for_missing(tmp_db_path):
    store = ResultStore(tmp_db_path)
    store.initialize()
    assert store.get_run(999) is None
