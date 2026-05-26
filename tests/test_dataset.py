import pytest
from pathlib import Path
import yaml
from mcpeval.dataset import load_suite, EvalSuite, MockToolDef, GraphStep, EvaluatorConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_suite_minimal():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert isinstance(suite, EvalSuite)
    assert suite.name == "Minimal Test Suite"
    assert suite.model == "claude-3-5-haiku-20241022"
    assert suite.mcp_server == "test-mcp"
    assert len(suite.cases) == 1


def test_load_suite_mock_tools():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert len(suite.mock_tools) == 2
    get_logs = next(t for t in suite.mock_tools if t.name == "get_logs")
    assert "logs" in get_logs.returns
    assert isinstance(get_logs.returns["logs"], list)


def test_load_suite_graph_steps():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    assert len(steps) >= 1
    assert steps[0].tool == "get_logs"


def test_load_suite_step_params_contain():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    step = suite.cases[0].expected_graph.steps[0]
    assert step.params_contain == {"service": "payment"}


def test_load_suite_optional_step():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    optional_step = next((s for s in steps if s.optional), None)
    assert optional_step is not None
    assert optional_step.tool == "notify_oncall"


def test_load_suite_must_follow():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    steps = suite.cases[0].expected_graph.steps
    run_playbook = next((s for s in steps if s.tool == "run_playbook"), None)
    assert run_playbook is not None
    assert run_playbook.must_follow == "get_logs"


def test_load_suite_max_steps():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert suite.cases[0].expected_graph.max_steps == 6


def test_load_suite_must_terminate():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    assert suite.cases[0].expected_graph.must_terminate is True


def test_load_suite_evaluators():
    suite = load_suite(FIXTURES_DIR / "minimal_suite.yaml")
    evals = suite.cases[0].evaluators
    assert len(evals) >= 1
    assert evals[0].type == "graph_match"
    assert evals[0].strict is False


def test_load_suite_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_suite("/nonexistent/path/suite.yaml")


def test_load_suite_invalid_yaml(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("name: [unclosed")
    with pytest.raises(Exception):
        load_suite(bad_file)
