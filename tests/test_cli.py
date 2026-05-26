# tests/test_cli.py
from __future__ import annotations
import dataclasses
from unittest.mock import AsyncMock, patch
from click.testing import CliRunner
from mcpeval.cli import cli
from mcpeval.runner import RunResult, CaseResult
import asyncio


def test_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "mcpeval" in result.output.lower()


def test_run_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "suite_file" in result.output.lower() or "SUITE_FILE" in result.output


def test_run_missing_file_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(tmp_path / "no_such.yaml")])
    assert result.exit_code != 0
    assert "no_such.yaml" in result.output


def _make_run_result(score: float) -> RunResult:
    return RunResult(
        run_id=None,
        eval_suite="Test Suite",
        model="claude-haiku-4-5-20251001",
        total_cases=1,
        passed=1 if score >= 0.5 else 0,
        failed=0 if score >= 0.5 else 1,
        overall_score=score,
        case_results=[
            CaseResult(
                case_id="c1",
                passed=score >= 0.5,
                tool_calls_made=[],
                tool_calls_expected=[],
                graph_match_score=score,
                llm_judge_score=None,
                rule_score=None,
                steps_taken=1,
                terminated_cleanly=True,
                raw_output="done",
            )
        ],
    )


def _make_suite_yaml(tmp_path) -> str:
    p = tmp_path / "suite.yaml"
    p.write_text(
        "name: Test Suite\n"
        "model: claude-haiku-4-5-20251001\n"
        "mcp_server: test\n"
        "mock_tools:\n"
        "  dummy_tool:\n"
        "    returns: {}\n"
        "cases:\n"
        "  - id: c1\n"
        "    input: do something\n"
        "    expected_graph:\n"
        "      steps:\n"
        "        - tool: dummy_tool\n"
    )
    return str(p)


def test_run_exits_zero_when_score_meets_threshold(tmp_path):
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = AsyncMock(return_value=_make_run_result(1.0))
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(cli, ["run", suite_path, "--threshold", "0.8", "--db", ":memory:"])
    assert result.exit_code == 0


def test_run_exits_one_when_score_below_threshold(tmp_path):
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = AsyncMock(return_value=_make_run_result(0.5))
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(cli, ["run", suite_path, "--threshold", "0.8", "--db", ":memory:"])
    assert result.exit_code == 1


def test_run_exits_zero_without_threshold(tmp_path):
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = AsyncMock(return_value=_make_run_result(0.0))
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(cli, ["run", suite_path, "--db", ":memory:"])
    assert result.exit_code == 0


def test_run_models_flag_runs_per_model(tmp_path):
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    call_count = 0

    async def fake_run_suite(suite):
        nonlocal call_count
        call_count += 1
        return _make_run_result(1.0)

    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = fake_run_suite
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(
                cli,
                ["run", suite_path, "--models", "haiku,sonnet", "--db", ":memory:"],
            )
    assert result.exit_code == 0
    assert call_count == 2


def test_run_models_alias_resolved(tmp_path):
    """haiku/sonnet/opus shorthand aliases expand to full model IDs."""
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    seen_models: list[str] = []

    async def fake_run_suite(suite):
        seen_models.append(suite.model)
        return _make_run_result(1.0)

    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = fake_run_suite
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(cli, ["run", suite_path, "--models", "haiku,sonnet", "--db", ":memory:"])
    assert result.exit_code == 0
    assert seen_models == ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"]


def test_run_models_exits_one_when_worst_score_below_threshold(tmp_path):
    runner = CliRunner()
    suite_path = _make_suite_yaml(tmp_path)
    scores = [1.0, 0.4]
    score_index = 0

    async def fake_run_suite(suite):
        nonlocal score_index
        result = _make_run_result(scores[score_index])
        score_index += 1
        return result

    with patch("mcpeval.cli.EvalRunner") as MockRunner:
        mock_instance = MockRunner.return_value
        mock_instance.run_suite = fake_run_suite
        with patch("mcpeval.cli.ResultStore"):
            result = runner.invoke(
                cli,
                ["run", suite_path, "--models", "haiku,sonnet", "--threshold", "0.8", "--db", ":memory:"],
            )
    assert result.exit_code == 1
