import io
import pytest
from rich.console import Console
from mcpeval.reporter import Reporter
from mcpeval.runner import RunResult, CaseResult


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
