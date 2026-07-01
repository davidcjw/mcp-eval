import pytest

from mcpeval.capture import ToolCallRecord
from mcpeval.evaluators.graph_match import GraphMatchEvaluator
from mcpeval.graph import Step, ToolCallGraph


def _record(tool_name: str, arguments: dict | None = None, ts: float = 0.0) -> ToolCallRecord:
    return ToolCallRecord(tool_name=tool_name, arguments=arguments or {}, timestamp=ts)


def test_perfect_match():
    graph = ToolCallGraph(steps=[Step("get_logs"), Step("run_playbook")])
    calls = [_record("get_logs"), _record("run_playbook")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.passed is True
    assert result.missing_required == []


def test_missing_required_step():
    graph = ToolCallGraph(steps=[Step("get_logs"), Step("run_playbook")])
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score < 1.0
    assert "run_playbook" in result.missing_required


def test_optional_step_absent_still_passes():
    graph = ToolCallGraph(
        steps=[
            Step("get_logs"),
            Step("notify_oncall", optional=True),
        ]
    )
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.passed is True


def test_params_contain_match():
    graph = ToolCallGraph(steps=[Step("get_logs", params_contain={"service": "payment"})])
    calls = [_record("get_logs", {"service": "payment", "extra": "ignored"})]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.missing_required == []


def test_params_contain_mismatch():
    graph = ToolCallGraph(steps=[Step("get_logs", params_contain={"service": "payment"})])
    calls = [_record("get_logs", {"service": "database"})]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score < 1.0
    assert "get_logs" in result.missing_required


def test_must_follow_respected():
    graph = ToolCallGraph(
        steps=[
            Step("get_logs"),
            Step("run_playbook", must_follow="get_logs"),
        ]
    )
    calls = [_record("get_logs", ts=0.0), _record("run_playbook", ts=1.0)]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0
    assert result.ordering_violations == []


def test_must_follow_violated():
    graph = ToolCallGraph(
        steps=[
            Step("get_logs"),
            Step("run_playbook", must_follow="get_logs"),
        ]
    )
    # run_playbook appears before get_logs
    calls = [_record("run_playbook", ts=0.0), _record("get_logs", ts=1.0)]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert len(result.ordering_violations) > 0
    assert result.score < 1.0


def test_max_steps_exceeded_strict():
    graph = ToolCallGraph(steps=[Step("get_logs")], max_steps=1)
    calls = [_record("get_logs"), _record("extra_call"), _record("another_call")]
    evaluator = GraphMatchEvaluator(strict=True)
    result = evaluator.evaluate(calls, graph, terminated_cleanly=True)
    assert result.steps_taken == 3
    assert result.score < 1.0


def test_terminate_penalty():
    graph = ToolCallGraph(steps=[Step("get_logs")], must_terminate=True)
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=False)
    assert result.score < 1.0


def test_score_never_below_zero():
    graph = ToolCallGraph(
        steps=[
            Step("a"),
            Step("b"),
            Step("c"),
            Step("d"),
        ],
        must_terminate=True,
    )
    calls = []
    result = GraphMatchEvaluator(strict=True).evaluate(calls, graph, terminated_cleanly=False)
    assert result.score >= 0.0


def test_score_never_above_one():
    graph = ToolCallGraph(steps=[Step("get_logs")])
    calls = [_record("get_logs")]
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score <= 1.0


def test_all_optional_no_calls():
    graph = ToolCallGraph(
        steps=[
            Step("a", optional=True),
            Step("b", optional=True),
        ]
    )
    calls = []
    result = GraphMatchEvaluator().evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == 1.0


def test_non_strict_fails_at_0_5():
    # 1 of 2 required steps present → base_score = 0.5, no penalties → 0.5 < 0.7 → failed
    graph = ToolCallGraph(steps=[Step("a"), Step("b")])
    calls = [_record("a")]
    result = GraphMatchEvaluator(strict=False).evaluate(calls, graph, terminated_cleanly=True)
    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_strict_requires_score_1():
    graph = ToolCallGraph(steps=[Step("a"), Step("b")])
    calls = [_record("a")]
    result = GraphMatchEvaluator(strict=True).evaluate(calls, graph, terminated_cleanly=True)
    assert result.passed is False
