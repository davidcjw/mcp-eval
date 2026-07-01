from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcpeval.capture import ToolCallRecord
from mcpeval.graph import Step, ToolCallGraph


@dataclass
class GraphMatchResult:
    score: float
    matched_steps: list[str]
    missing_required: list[str]
    ordering_violations: list[str]
    params_mismatches: list[str]
    steps_taken: int
    terminated_cleanly: bool
    passed: bool


def _params_match(actual_args: dict[str, Any], params_contain: dict[str, Any]) -> bool:
    return all(actual_args.get(k) == v for k, v in params_contain.items())


def _find_matching_call_index(
    calls: list[ToolCallRecord],
    step: Step,
) -> int | None:
    for i, call in enumerate(calls):
        if call.tool_name != step.tool:
            continue
        if step.params_contain and not _params_match(call.arguments, step.params_contain):
            continue
        return i
    return None


class GraphMatchEvaluator:
    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def evaluate(
        self,
        actual_calls: list[ToolCallRecord],
        expected: ToolCallGraph,
        terminated_cleanly: bool,
    ) -> GraphMatchResult:
        matched_steps: list[str] = []
        missing_required: list[str] = []
        ordering_violations: list[str] = []
        params_mismatches: list[str] = []

        required_steps = expected.required_steps
        last_match_index: dict[str, int] = {}

        for step in required_steps:
            idx = _find_matching_call_index(actual_calls, step)
            if idx is None:
                missing_required.append(step.tool)
                continue

            matched_steps.append(step.tool)

            if step.must_follow:
                predecessor_idx = last_match_index.get(step.must_follow)
                if predecessor_idx is None or predecessor_idx >= idx:
                    ordering_violations.append(step.tool)

            last_match_index[step.tool] = idx

        total_required = len(required_steps)
        if total_required == 0:
            base_score = 1.0
        else:
            matched_count = len(matched_steps) - len(ordering_violations)
            base_score = matched_count / total_required

        penalty = 0.0
        penalty += len(ordering_violations) * 0.1

        if expected.must_terminate and not terminated_cleanly:
            penalty += 0.2

        steps_taken = len(actual_calls)
        if self.strict and steps_taken > expected.max_steps:
            excess = steps_taken - expected.max_steps
            penalty += (excess / max(expected.max_steps, 1)) * 0.1

        score = max(0.0, min(1.0, base_score - penalty))
        pass_threshold = 1.0 if self.strict else 0.7
        passed = score >= pass_threshold

        return GraphMatchResult(
            score=score,
            matched_steps=matched_steps,
            missing_required=missing_required,
            ordering_violations=ordering_violations,
            params_mismatches=params_mismatches,
            steps_taken=steps_taken,
            terminated_cleanly=terminated_cleanly,
            passed=passed,
        )
