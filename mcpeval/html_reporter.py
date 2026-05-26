# mcpeval/html_reporter.py
from __future__ import annotations
import html
from pathlib import Path
from datetime import datetime, timezone

from mcpeval.runner import RunResult, CaseResult

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
.summary { display: flex; gap: 2rem; margin-bottom: 1.5rem; }
.stat { text-align: center; }
.stat .value { font-size: 2rem; font-weight: bold; }
.stat .label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 2px solid #ddd; font-size: 0.85rem; text-transform: uppercase; color: #666; }
td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }
.pass { color: #16a34a; font-weight: bold; }
.fail { color: #dc2626; font-weight: bold; }
.score-high { color: #16a34a; }
.score-mid { color: #d97706; }
.score-low { color: #dc2626; }
.error { color: #dc2626; font-size: 0.8rem; font-family: monospace; }
.section-title { font-size: 1.1rem; font-weight: bold; margin: 1.5rem 0 0.5rem; }
"""


def _score_class(score: float) -> str:
    if score >= 0.8:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def _case_rows(case_results: list[CaseResult]) -> str:
    rows = []
    for cr in case_results:
        status_cls = "pass" if cr.passed else "fail"
        status_txt = "PASS" if cr.passed else "FAIL"
        score_cls = _score_class(cr.graph_match_score)
        rule = f"{cr.rule_score:.2f}" if cr.rule_score is not None else "—"
        judge = f"{cr.llm_judge_score:.2f}" if cr.llm_judge_score is not None else "—"
        error_cell = f'<span class="error">{html.escape(str(cr.error))}</span>' if cr.error else ""
        rows.append(
            f"<tr>"
            f"<td>{html.escape(str(cr.case_id))}</td>"
            f'<td class="{status_cls}">{status_txt}</td>'
            f'<td class="{score_cls}">{cr.graph_match_score:.2f}</td>'
            f"<td>{rule}</td>"
            f"<td>{judge}</td>"
            f"<td>{cr.steps_taken}</td>"
            f"<td>{'✓' if cr.terminated_cleanly else '✗'}</td>"
            f"<td>{error_cell}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _case_table(case_results: list[CaseResult]) -> str:
    return (
        "<table>"
        "<thead><tr>"
        "<th>Case ID</th><th>Status</th><th>Graph Score</th>"
        "<th>Rule Score</th><th>LLM Judge</th><th>Steps</th>"
        "<th>Terminated</th><th>Error</th>"
        "</tr></thead>"
        f"<tbody>{_case_rows(case_results)}</tbody>"
        "</table>"
    )


class HtmlReporter:
    def write_report(self, result: RunResult, path: str | Path) -> None:
        Path(path).write_text(self._render_single(result), encoding="utf-8")

    def write_multi_model_report(self, results: list[RunResult], path: str | Path) -> None:
        Path(path).write_text(self._render_multi(results), encoding="utf-8")

    def _render_single(self, result: RunResult) -> str:
        score_cls = _score_class(result.overall_score)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body = (
            f"<h1>{html.escape(result.eval_suite)}</h1>"
            f'<div class="meta">Model: {html.escape(result.model)} &nbsp;|&nbsp; Generated: {now}'
            + (f" &nbsp;|&nbsp; Run ID: {result.run_id}" if result.run_id else "")
            + "</div>"
            f'<div class="summary">'
            f'<div class="stat"><div class="value">{result.total_cases}</div><div class="label">Total</div></div>'
            f'<div class="stat"><div class="value" style="color:#16a34a">{result.passed}</div><div class="label">Passed</div></div>'
            f'<div class="stat"><div class="value" style="color:#dc2626">{result.failed}</div><div class="label">Failed</div></div>'
            f'<div class="stat"><div class="value {score_cls}">{result.overall_score:.2f}</div><div class="label">Score</div></div>'
            f"</div>"
            f'<div class="section-title">Cases</div>'
            + _case_table(result.case_results)
        )
        return _html_page(html.escape(result.eval_suite), body)

    def _render_multi(self, results: list[RunResult]) -> str:
        suite_name = html.escape(results[0].eval_suite) if results else "Eval Results"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        overview_rows = "\n".join(
            f"<tr>"
            f"<td>{html.escape(r.model)}</td>"
            f"<td>{r.passed}/{r.total_cases}</td>"
            f'<td class="{_score_class(r.overall_score)}">{r.overall_score:.2f}</td>'
            f"</tr>"
            for r in results
        )
        overview = (
            "<table><thead><tr><th>Model</th><th>Passed</th><th>Score</th></tr></thead>"
            f"<tbody>{overview_rows}</tbody></table>"
        )
        per_model = ""
        for r in results:
            per_model += (
                f'<div class="section-title">{html.escape(r.model)}</div>'
                + _case_table(r.case_results)
            )
        body = (
            f"<h1>{suite_name} — Multi-Model Comparison</h1>"
            f'<div class="meta">Generated: {now}</div>'
            f'<div class="section-title">Overview</div>'
            + overview
            + per_model
        )
        return _html_page(f"{suite_name} — Multi-Model", body)


def _html_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head>"
        f"<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style>"
        f"</head><body>{body}</body></html>"
    )
