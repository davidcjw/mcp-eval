from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RuleResult:
    score: float
    passed: bool
    violations: list[str] = field(default_factory=list)


class RuleEvaluator:
    def __init__(
        self,
        checks: list[dict],
        strict: bool = False,
        threshold: float = 0.7,
    ) -> None:
        self._checks = checks
        self._threshold = 1.0 if strict else threshold

    def evaluate(self, raw_output: str) -> RuleResult:
        violations: list[str] = []
        for check in self._checks:
            if "contains" in check:
                if check["contains"] not in raw_output:
                    violations.append(f"missing: {check['contains']!r}")
            elif "not_contains" in check:
                if check["not_contains"] in raw_output:
                    violations.append(f"unexpected: {check['not_contains']!r}")
            elif "regex" in check:
                if not re.search(check["regex"], raw_output):
                    violations.append(f"regex not matched: {check['regex']!r}")

        total = len(self._checks)
        score = (total - len(violations)) / total if total > 0 else 1.0
        return RuleResult(
            score=score,
            passed=score >= self._threshold,
            violations=violations,
        )
