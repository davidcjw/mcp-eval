import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcpeval.runner import EvalRunner, RunResult, CaseResult
from mcpeval.dataset import EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig


def _make_suite(cases: list[Case] | None = None) -> EvalSuite:
    if cases is None:
        cases = [Case(
            id="c001",
            input="Do the thing",
            expected_graph=ExpectedGraph(
                steps=[GraphStep(tool="get_logs")],
                max_steps=5,
                must_terminate=True,
            ),
            evaluators=[EvaluatorConfig(type="graph_match", strict=False)],
        )]
    return EvalSuite(
        name="Test Suite",
        model="claude-3-5-haiku-20241022",
        mcp_server="test-mcp",
        mock_tools=[MockToolDef(name="get_logs", returns={"logs": []})],
        cases=cases,
    )


def _mock_end_turn_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [MagicMock(type="text", text="All done.")]
    return response


def _mock_tool_use_then_end(tool_name: str, tool_input: dict):
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_use_123"
    tool_block.name = tool_name
    tool_block.input = tool_input

    response1 = MagicMock()
    response1.stop_reason = "tool_use"
    response1.content = [tool_block]

    response2 = MagicMock()
    response2.stop_reason = "end_turn"
    response2.content = [MagicMock(type="text", text="Done.")]

    return [response1, response2]


@pytest.mark.asyncio
async def test_run_suite_returns_run_result():
    runner = EvalRunner(anthropic_api_key="test-key")

    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(_make_suite())

    assert isinstance(result, RunResult)
    assert result.eval_suite == "Test Suite"
    assert result.total_cases == 1
    assert result.model == "claude-3-5-haiku-20241022"


@pytest.mark.asyncio
async def test_run_case_terminated_cleanly_true():
    runner = EvalRunner(anthropic_api_key="test-key")

    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(_make_suite())

    assert result.case_results[0].terminated_cleanly is True


@pytest.mark.asyncio
async def test_run_case_terminated_cleanly_false():
    runner = EvalRunner(anthropic_api_key="test-key", max_agent_turns=1)

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu_1"
    tool_block.name = "get_logs"
    tool_block.input = {}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]

    mock_create = AsyncMock(return_value=response)
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(_make_suite())

    assert result.case_results[0].terminated_cleanly is False


@pytest.mark.asyncio
async def test_run_case_captures_tool_calls():
    runner = EvalRunner(anthropic_api_key="test-key")
    responses = _mock_tool_use_then_end("get_logs", {})

    mock_create = AsyncMock(side_effect=responses)
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(_make_suite())

    cr = result.case_results[0]
    assert len(cr.tool_calls_made) >= 1
    assert cr.tool_calls_made[0]["tool_name"] == "get_logs"


@pytest.mark.asyncio
async def test_run_suite_aggregates_scores():
    case_a = Case(
        id="a",
        input="do a",
        expected_graph=ExpectedGraph(steps=[GraphStep(tool="get_logs")], must_terminate=True),
        evaluators=[EvaluatorConfig(type="graph_match")],
    )
    case_b = Case(
        id="b",
        input="do b",
        expected_graph=ExpectedGraph(steps=[GraphStep(tool="run_playbook")], must_terminate=True),
        evaluators=[EvaluatorConfig(type="graph_match")],
    )
    suite = EvalSuite(
        name="Multi",
        model="claude-3-5-haiku-20241022",
        mcp_server="test",
        mock_tools=[
            MockToolDef(name="get_logs", returns={}),
            MockToolDef(name="run_playbook", returns={}),
        ],
        cases=[case_a, case_b],
    )
    runner = EvalRunner(anthropic_api_key="test-key")

    responses_a = _mock_tool_use_then_end("get_logs", {})
    responses_b = [_mock_end_turn_response()]
    all_responses = responses_a + responses_b

    call_idx = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_idx
        r = all_responses[min(call_idx, len(all_responses) - 1)]
        call_idx += 1
        return r

    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(side_effect=side_effect)
        result = await runner.run_suite(suite)

    assert result.total_cases == 2
    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.asyncio
async def test_runner_saves_to_store(tmp_db_path):
    from mcpeval.store import ResultStore
    store = ResultStore(tmp_db_path)
    store.initialize()

    runner = EvalRunner(store=store, anthropic_api_key="test-key")
    mock_create = AsyncMock(return_value=_mock_end_turn_response())
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = mock_create
        result = await runner.run_suite(_make_suite())

    assert result.run_id is not None
    stored = store.get_run(result.run_id)
    assert stored is not None
    assert stored["eval_suite"] == "Test Suite"


@pytest.mark.asyncio
async def test_run_suite_isolates_case_errors():
    """A crashing case does not prevent subsequent cases from running."""
    from mcpeval.dataset import EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig
    cases = [
        Case(
            id="crash",
            input="trigger crash",
            expected_graph=ExpectedGraph(steps=[], must_terminate=True),
            evaluators=[],
        ),
        Case(
            id="normal",
            input="Do the thing",
            expected_graph=ExpectedGraph(steps=[GraphStep(tool="get_logs")], must_terminate=True),
            evaluators=[EvaluatorConfig(type="graph_match")],
        ),
    ]
    suite = EvalSuite(
        name="Isolation Test",
        model="claude-3-5-haiku-20241022",
        mcp_server="test",
        mock_tools=[MockToolDef(name="get_logs", returns={})],
        cases=cases,
    )

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated API failure")
        return _mock_end_turn_response()

    runner = EvalRunner(anthropic_api_key="test-key")
    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(side_effect=side_effect)
        result = await runner.run_suite(suite)

    assert result.total_cases == 2
    crash_cr = next(cr for cr in result.case_results if cr.case_id == "crash")
    normal_cr = next(cr for cr in result.case_results if cr.case_id == "normal")
    assert crash_cr.passed is False
    assert crash_cr.error is not None
    assert normal_cr.error is None


@pytest.mark.asyncio
async def test_run_case_wires_rule_evaluator():
    from mcpeval.dataset import EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig

    suite = EvalSuite(
        name="Rule Test",
        model="claude-3-5-haiku-20241022",
        mcp_server="test",
        mock_tools=[MockToolDef(name="get_logs", returns={})],
        cases=[Case(
            id="rule_case",
            input="Do the thing",
            expected_graph=ExpectedGraph(steps=[GraphStep(tool="get_logs")], must_terminate=True),
            evaluators=[
                EvaluatorConfig(type="graph_match"),
                EvaluatorConfig(type="rule", checks=[{"contains": "done"}], threshold=0.7),
            ],
        )],
    )
    runner = EvalRunner(anthropic_api_key="test-key")

    responses = _mock_tool_use_then_end("get_logs", {})
    # Override last response to include "done" in text
    responses[-1].content = [MagicMock(type="text", text="All done.")]

    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(side_effect=responses)
        result = await runner.run_suite(suite)

    cr = result.case_results[0]
    assert cr.rule_score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_case_rule_failure_marks_not_passed():
    from mcpeval.dataset import EvalSuite, MockToolDef, Case, ExpectedGraph, GraphStep, EvaluatorConfig

    suite = EvalSuite(
        name="Rule Fail Test",
        model="claude-3-5-haiku-20241022",
        mcp_server="test",
        mock_tools=[MockToolDef(name="get_logs", returns={})],
        cases=[Case(
            id="rule_fail",
            input="Do the thing",
            expected_graph=ExpectedGraph(steps=[GraphStep(tool="get_logs")], must_terminate=True),
            evaluators=[
                EvaluatorConfig(type="graph_match"),
                EvaluatorConfig(type="rule", checks=[{"contains": "IMPOSSIBLE_STRING_XYZ"}]),
            ],
        )],
    )
    runner = EvalRunner(anthropic_api_key="test-key")
    responses = _mock_tool_use_then_end("get_logs", {})

    with patch("mcpeval.runner.AsyncAnthropic") as mock_anthropic:
        mock_anthropic.return_value.messages.create = AsyncMock(side_effect=responses)
        result = await runner.run_suite(suite)

    cr = result.case_results[0]
    assert cr.rule_score == pytest.approx(0.0)
    assert cr.passed is False
