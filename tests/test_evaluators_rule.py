import pytest
from mcpeval.evaluators.rule import RuleEvaluator, RuleResult


def test_contains_passes():
    result = RuleEvaluator(checks=[{"contains": "resolved"}]).evaluate("Issue resolved.")
    assert result.score == 1.0
    assert result.passed is True
    assert result.violations == []


def test_contains_fails():
    result = RuleEvaluator(checks=[{"contains": "resolved"}]).evaluate("Still open.")
    assert result.score == 0.0
    assert result.passed is False
    assert len(result.violations) == 1


def test_not_contains_passes():
    result = RuleEvaluator(checks=[{"not_contains": "error"}]).evaluate("All clear.")
    assert result.score == 1.0
    assert result.passed is True


def test_not_contains_fails():
    result = RuleEvaluator(checks=[{"not_contains": "error"}]).evaluate("Fatal error.")
    assert result.score == 0.0
    assert result.passed is False


def test_regex_passes():
    result = RuleEvaluator(checks=[{"regex": r"ticket #\d+"}]).evaluate("See ticket #42.")
    assert result.score == 1.0
    assert result.passed is True


def test_regex_fails():
    result = RuleEvaluator(checks=[{"regex": r"ticket #\d+"}]).evaluate("No ticket here.")
    assert result.score == 0.0
    assert result.passed is False


def test_partial_score_below_threshold():
    result = RuleEvaluator(
        checks=[{"contains": "resolved"}, {"contains": "ticket"}]
    ).evaluate("Issue resolved.")
    assert result.score == pytest.approx(0.5)
    assert result.passed is False  # 0.5 < default threshold 0.7


def test_custom_threshold():
    result = RuleEvaluator(
        checks=[{"contains": "a"}, {"contains": "b"}], threshold=0.5
    ).evaluate("only a here")
    assert result.score == pytest.approx(0.5)
    assert result.passed is True  # 0.5 >= 0.5


def test_strict_requires_all_checks():
    result = RuleEvaluator(
        checks=[{"contains": "a"}, {"contains": "b"}], strict=True
    ).evaluate("only a here")
    assert result.passed is False


def test_strict_all_pass():
    result = RuleEvaluator(
        checks=[{"contains": "a"}, {"contains": "b"}], strict=True
    ).evaluate("a and b here")
    assert result.passed is True


def test_empty_checks_always_passes():
    result = RuleEvaluator(checks=[]).evaluate("anything")
    assert result.score == 1.0
    assert result.passed is True


def test_multiple_violations_recorded():
    result = RuleEvaluator(
        checks=[{"contains": "x"}, {"contains": "y"}, {"contains": "z"}]
    ).evaluate("nothing here")
    assert len(result.violations) == 3
