# tests/test_html_reporter.py
from __future__ import annotations

from mcpeval.html_reporter import HtmlReporter
from mcpeval.runner import CaseResult, RunResult


def _make_case_result(case_id: str, passed: bool, score: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=passed,
        tool_calls_made=[{"tool_name": "get_logs", "arguments": {}}],
        tool_calls_expected=[{"tool": "get_logs"}],
        graph_match_score=score,
        llm_judge_score=0.9 if passed else None,
        rule_score=1.0 if passed else None,
        steps_taken=2,
        terminated_cleanly=passed,
        raw_output="All done." if passed else "",
        error=None if passed else "timeout",
    )


def _make_run_result(model: str, score: float) -> RunResult:
    cr = _make_case_result("c1", score >= 0.5, score)
    return RunResult(
        run_id=1,
        eval_suite="Incident Response Suite",
        model=model,
        total_cases=1,
        passed=1 if score >= 0.5 else 0,
        failed=0 if score >= 0.5 else 1,
        overall_score=score,
        case_results=[cr],
    )


def test_write_report_creates_file(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 1.0), out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_single_report_contains_suite_name(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 1.0), out)
    html = out.read_text()
    assert "Incident Response Suite" in html


def test_single_report_contains_model(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 1.0), out)
    html = out.read_text()
    assert "claude-haiku-4-5-20251001" in html


def test_single_report_contains_case_id(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 1.0), out)
    html = out.read_text()
    assert "c1" in html


def test_single_report_shows_pass_fail(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 0.0), out)
    html = out.read_text()
    assert "FAIL" in html


def test_single_report_shows_error(tmp_path):
    out = tmp_path / "report.html"
    HtmlReporter().write_report(_make_run_result("claude-haiku-4-5-20251001", 0.0), out)
    html = out.read_text()
    assert "timeout" in html


def test_multi_model_report_contains_all_models(tmp_path):
    out = tmp_path / "multi.html"
    results = [
        _make_run_result("claude-haiku-4-5-20251001", 1.0),
        _make_run_result("claude-sonnet-4-6", 0.5),
    ]
    HtmlReporter().write_multi_model_report(results, out)
    html = out.read_text()
    assert "claude-haiku-4-5-20251001" in html
    assert "claude-sonnet-4-6" in html


def test_multi_model_report_shows_scores(tmp_path):
    out = tmp_path / "multi.html"
    results = [
        _make_run_result("claude-haiku-4-5-20251001", 1.0),
        _make_run_result("claude-sonnet-4-6", 0.5),
    ]
    HtmlReporter().write_multi_model_report(results, out)
    html = out.read_text()
    assert "1.00" in html
    assert "0.50" in html
