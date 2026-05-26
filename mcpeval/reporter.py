from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich import box

from mcpeval.runner import CaseResult, RunResult


class Reporter:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def print_run_summary(self, result: RunResult) -> None:
        self._console.print(f"\n[bold]Eval Suite:[/bold] {result.eval_suite}")
        self._console.print(f"[bold]Model:[/bold] {result.model}")
        self._console.print(
            f"[bold]Results:[/bold] {result.passed}/{result.total_cases} passed  "
            f"| Overall score: {result.overall_score:.2f}"
        )

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Case ID", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Terminated", justify="center")

        for cr in result.case_results:
            status = "[green]PASS[/green]" if cr.passed else "[red]FAIL[/red]"
            table.add_row(
                cr.case_id,
                status,
                f"{cr.graph_match_score:.2f}",
                str(cr.steps_taken),
                "✓" if cr.terminated_cleanly else "✗",
            )

        self._console.print(table)

    def print_multi_model_summary(self, results: list[RunResult]) -> None:
        self._console.print("\n[bold]Multi-Model Comparison[/bold]")
        self._console.print(f"[bold]Suite:[/bold] {results[0].eval_suite if results else ''}")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Passed", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Score", justify="right")

        for result in results:
            score_style = "green" if result.overall_score >= 0.8 else "yellow" if result.overall_score >= 0.5 else "red"
            table.add_row(
                result.model,
                str(result.passed),
                str(result.failed),
                f"[{score_style}]{result.overall_score:.2f}[/{score_style}]",
            )

        self._console.print(table)

    def print_case_detail(self, cr: CaseResult) -> None:
        status = "[green]PASS[/green]" if cr.passed else "[red]FAIL[/red]"
        self._console.print(f"\n[bold]Case:[/bold] {cr.case_id}  {status}")
        self._console.print(f"  Score: {cr.graph_match_score:.2f}")
        self._console.print(f"  Steps taken: {cr.steps_taken}")
        self._console.print(f"  Terminated cleanly: {cr.terminated_cleanly}")
        if cr.tool_calls_made:
            self._console.print(f"  Tool calls: {[c['tool_name'] for c in cr.tool_calls_made]}")
        if cr.raw_output:
            self._console.print(f"  Output: {cr.raw_output[:200]}")
