from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    tool_calls_made: list[dict[str, Any]]
    tool_calls_expected: list[dict[str, Any]]
    graph_match_score: float
    llm_judge_score: float | None
    steps_taken: int
    terminated_cleanly: bool
    raw_output: str


@dataclass
class RunResult:
    run_id: str | None
    eval_suite: str
    model: str
    total_cases: int
    passed: int
    failed: int
    overall_score: float
    case_results: list[CaseResult] = field(default_factory=list)
