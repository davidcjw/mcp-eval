# mcpeval/cli.py
from __future__ import annotations
import sys
import asyncio
import dataclasses

import click

from mcpeval.dataset import load_suite
from mcpeval.runner import EvalRunner, RunResult
from mcpeval.store import ResultStore
from mcpeval.reporter import Reporter

_MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


@click.group()
def cli() -> None:
    """mcpeval — MCP tool-call evaluation harness."""


@cli.command()
@click.argument("suite_file")
@click.option("--threshold", type=float, default=None, help="Exit 1 if overall_score < threshold.")
@click.option("--models", default=None, help="Comma-separated model IDs or shorthand aliases (haiku/sonnet/opus).")
@click.option("--output", default=None, help="Write HTML report to this path.")
@click.option("--db", default="mcpeval.db", show_default=True, help="SQLite DB path.")
def run(
    suite_file: str,
    threshold: float | None,
    models: str | None,
    output: str | None,
    db: str,
) -> None:
    """Run an eval suite from SUITE_FILE (YAML)."""
    try:
        suite = load_suite(suite_file)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    store = ResultStore(db)
    store.initialize()
    runner = EvalRunner(store=store)
    reporter = Reporter()

    if models:
        model_list = [_MODEL_ALIASES.get(m.strip(), m.strip()) for m in models.split(",")]
        run_results: list[RunResult] = []
        for model in model_list:
            model_suite = dataclasses.replace(suite, model=model)
            run_results.append(asyncio.run(runner.run_suite(model_suite)))
        reporter.print_multi_model_summary(run_results)
        if output:
            from mcpeval.html_reporter import HtmlReporter
            HtmlReporter().write_multi_model_report(run_results, output)
        if threshold is not None:
            worst = min(r.overall_score for r in run_results)
            if worst < threshold:
                sys.exit(1)
    else:
        run_result = asyncio.run(runner.run_suite(suite))
        reporter.print_run_summary(run_result)
        if output:
            from mcpeval.html_reporter import HtmlReporter
            HtmlReporter().write_report(run_result, output)
        if threshold is not None and run_result.overall_score < threshold:
            sys.exit(1)
