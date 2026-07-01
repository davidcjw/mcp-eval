import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpeval.capture import ToolCallRecord
from mcpeval.evaluators.llm_judge import LLMJudgeEvaluator


def _mock_judge_response(score: float, reasoning: str):
    block = MagicMock()
    block.text = json.dumps({"score": score, "reasoning": reasoning})
    response = MagicMock()
    response.content = [block]
    return response


@pytest.mark.asyncio
async def test_evaluate_returns_score():
    judge = LLMJudgeEvaluator(
        criteria="Agent resolved the issue",
        model="claude-3-5-haiku-20241022",
        api_key="test-key",
    )
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_judge_response(0.9, "Agent correctly resolved.")
        )
        result = await judge.evaluate(
            raw_output="Issue resolved successfully.",
            task_input="Resolve the payment issue",
            tool_calls=[],
        )
    assert result.score == pytest.approx(0.9)
    assert result.passed is True
    assert result.reasoning == "Agent correctly resolved."


@pytest.mark.asyncio
async def test_evaluate_passed_false_below_threshold():
    judge = LLMJudgeEvaluator(
        criteria="Agent resolved the issue",
        model="claude-3-5-haiku-20241022",
        api_key="test-key",
        threshold=0.8,
    )
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_judge_response(0.6, "Only partial resolution.")
        )
        result = await judge.evaluate("Attempted.", "Resolve the issue", [])
    assert result.score == pytest.approx(0.6)
    assert result.passed is False


@pytest.mark.asyncio
async def test_evaluate_clamps_score_above_one():
    judge = LLMJudgeEvaluator(criteria="x", model="m", api_key="test-key")
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_judge_response(1.5, "over")
        )
        result = await judge.evaluate("output", "input", [])
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_evaluate_clamps_score_below_zero():
    judge = LLMJudgeEvaluator(criteria="x", model="m", api_key="test-key")
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_judge_response(-0.5, "negative")
        )
        result = await judge.evaluate("output", "input", [])
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_evaluate_handles_malformed_json():
    judge = LLMJudgeEvaluator(criteria="x", model="m", api_key="test-key")
    block = MagicMock()
    block.text = "I think it's pretty good, maybe 0.8"
    response = MagicMock()
    response.content = [block]
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(return_value=response)
        result = await judge.evaluate("output", "input", [])
    assert result.score == 0.0
    assert result.passed is False
    assert "Failed to parse" in result.reasoning


@pytest.mark.asyncio
async def test_evaluate_includes_tool_calls_in_prompt():
    judge = LLMJudgeEvaluator(criteria="check tool usage", model="m", api_key="test-key")
    calls = [
        ToolCallRecord(tool_name="get_logs", arguments={"service": "payment"}, timestamp=0.0),
        ToolCallRecord(tool_name="run_playbook", arguments={}, timestamp=1.0),
    ]
    captured: list[str] = []

    async def capture_call(*args, **kwargs):
        captured.append(kwargs["messages"][0]["content"])
        return _mock_judge_response(1.0, "good")

    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = capture_call
        await judge.evaluate("output", "input", calls)

    assert "get_logs" in captured[0]
    assert "run_playbook" in captured[0]


@pytest.mark.asyncio
async def test_evaluate_handles_markdown_wrapped_json():
    """Some models wrap JSON in ```json ... ``` code fences — must still parse."""
    judge = LLMJudgeEvaluator(criteria="x", model="m", api_key="test-key")
    import json as _json

    block = MagicMock()
    block.text = "```json\n" + _json.dumps({"score": 0.85, "reasoning": "Good job."}) + "\n```"
    response = MagicMock()
    response.content = [block]
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(return_value=response)
        result = await judge.evaluate("output", "input", [])
    assert result.score == pytest.approx(0.85)
    assert result.reasoning == "Good job."
    assert result.passed is True


@pytest.mark.asyncio
async def test_evaluate_default_threshold_is_0_7():
    judge = LLMJudgeEvaluator(criteria="x", model="m", api_key="test-key")
    with patch("mcpeval.evaluators.llm_judge.AsyncAnthropic") as mock_cls:
        mock_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_judge_response(0.7, "ok")
        )
        result = await judge.evaluate("output", "input", [])
    assert result.passed is True  # 0.7 >= 0.7
