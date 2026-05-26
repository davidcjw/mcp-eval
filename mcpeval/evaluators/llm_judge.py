from __future__ import annotations
import json
import os
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from mcpeval.capture import ToolCallRecord


_JUDGE_PROMPT = """\
You are an impartial evaluator for AI agent responses.

Task given to agent:
{task_input}

Agent's final response:
{raw_output}

Tool calls made by agent:
{tool_calls_summary}

Evaluation criteria:
{criteria}

Rate the agent's performance from 0.0 to 1.0:
- 1.0 = fully meets criteria
- 0.7 = mostly meets criteria with minor issues
- 0.5 = partially meets criteria
- 0.0 = completely fails to meet criteria

Respond with JSON only, no other text:
{{"score": <float 0.0-1.0>, "reasoning": "<one concise sentence>"}}\
"""


@dataclass
class LLMJudgeResult:
    score: float
    reasoning: str
    passed: bool


class LLMJudgeEvaluator:
    def __init__(
        self,
        criteria: str,
        model: str,
        api_key: str | None = None,
        threshold: float = 0.7,
    ) -> None:
        self._criteria = criteria
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._threshold = threshold

    async def evaluate(
        self,
        raw_output: str,
        task_input: str,
        tool_calls: list[ToolCallRecord],
    ) -> LLMJudgeResult:
        tool_calls_summary = "\n".join(
            f"- {c.tool_name}({json.dumps(c.arguments)})" for c in tool_calls
        ) or "(none)"

        prompt = _JUDGE_PROMPT.format(
            task_input=task_input,
            raw_output=raw_output or "(no text output)",
            tool_calls_summary=tool_calls_summary,
            criteria=self._criteria,
        )

        anthropic = AsyncAnthropic(api_key=self._api_key)
        response = await anthropic.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Strip markdown code fences that some models add (```json ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        try:
            data = json.loads(text)
            score = float(data["score"])
            reasoning = str(data.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 0.0
            reasoning = f"Failed to parse judge response: {text[:120]}"

        score = max(0.0, min(1.0, score))
        return LLMJudgeResult(score=score, reasoning=reasoning, passed=score >= self._threshold)
