import pytest
from pathlib import Path
from mcpeval.dataset import (
    EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig
)
from mcpeval.runner import RunResult, CaseResult


@pytest.fixture
def sample_mock_tools() -> list[MockToolDef]:
    return [
        MockToolDef(name="get_logs", returns={"logs": ["ERROR: timeout"]}),
        MockToolDef(name="run_playbook", returns={"status": "ok", "steps_completed": 3}),
        MockToolDef(name="notify_oncall", returns={"notified": True}),
    ]


@pytest.fixture
def sample_suite(sample_mock_tools) -> EvalSuite:
    return EvalSuite(
        name="Test Suite",
        model="claude-sonnet-4-20250514",
        mcp_server="test-mcp",
        mock_tools=sample_mock_tools,
        cases=[
            Case(
                id="test_001",
                input="Check logs and run playbook",
                expected_graph=ExpectedGraph(
                    steps=[
                        GraphStep(tool="get_logs", params_contain={"service": "payment"}),
                        GraphStep(tool="run_playbook", must_follow="get_logs"),
                        GraphStep(tool="notify_oncall", optional=True),
                    ],
                    max_steps=6,
                    must_terminate=True,
                ),
                evaluators=[EvaluatorConfig(type="graph_match", strict=False)],
            )
        ],
    )


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def passing_case_result() -> CaseResult:
    return CaseResult(
        case_id="test_001",
        passed=True,
        tool_calls_made=[{"tool_name": "get_logs", "arguments": {"service": "payment"}}],
        tool_calls_expected=[{"tool": "get_logs", "params_contain": {"service": "payment"}}],
        graph_match_score=1.0,
        llm_judge_score=None,
        steps_taken=2,
        terminated_cleanly=True,
        raw_output="Playbook executed successfully.",
    )


@pytest.fixture
def failing_case_result() -> CaseResult:
    return CaseResult(
        case_id="test_002",
        passed=False,
        tool_calls_made=[],
        tool_calls_expected=[{"tool": "get_logs"}],
        graph_match_score=0.0,
        llm_judge_score=None,
        steps_taken=0,
        terminated_cleanly=False,
        raw_output="",
    )


@pytest.fixture
def sample_run_result(passing_case_result, failing_case_result) -> RunResult:
    return RunResult(
        run_id=None,
        eval_suite="Test Suite",
        model="claude-sonnet-4-20250514",
        total_cases=2,
        passed=1,
        failed=1,
        overall_score=0.5,
        case_results=[passing_case_result, failing_case_result],
    )
