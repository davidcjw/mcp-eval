import io

from rich.console import Console

from mcpeval.reporter import Reporter
from mcpeval.runner import RunResult


def _make_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=True)
    return console, buf


def test_print_run_summary_renders(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert len(output) > 0
    assert "test_001" in output or "Test Suite" in output


def test_print_run_summary_shows_pass_fail_counts(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "1" in output  # at least one "1" for pass or fail count


def test_print_run_summary_shows_score(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "0.5" in output or "50" in output


def test_print_run_summary_shows_case_ids(sample_run_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_run_summary(sample_run_result)
    output = buf.getvalue()
    assert "test_001" in output
    assert "test_002" in output


def test_print_case_detail_shows_score(passing_case_result):
    console, buf = _make_console()
    reporter = Reporter(console=console)
    reporter.print_case_detail(passing_case_result)
    output = buf.getvalue()
    assert "test_001" in output
    assert "1.0" in output or "100" in output


def test_default_console_created_if_none():
    reporter = Reporter(console=None)
    assert reporter._console is not None


def test_print_multi_model_summary_shows_all_models():
    console, buf = _make_console()
    reporter = Reporter(console=console)
    results = [
        RunResult(
            run_id=None,
            eval_suite="Suite",
            model="claude-haiku-4-5-20251001",
            total_cases=2,
            passed=2,
            failed=0,
            overall_score=1.0,
        ),
        RunResult(
            run_id=None,
            eval_suite="Suite",
            model="claude-sonnet-4-6",
            total_cases=2,
            passed=1,
            failed=1,
            overall_score=0.5,
        ),
    ]
    reporter.print_multi_model_summary(results)
    output = buf.getvalue()
    assert "claude-haiku-4-5-20251001" in output
    assert "claude-sonnet-4-6" in output
    assert "1.00" in output
    assert "0.50" in output
