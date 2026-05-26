# tests/test_cli.py
from __future__ import annotations
from click.testing import CliRunner
from mcpeval.cli import cli


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
