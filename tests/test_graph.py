from mcpeval.dataset import ExpectedGraph
from mcpeval.dataset import GraphStep as DatasetStep
from mcpeval.graph import Step, ToolCallGraph


def test_step_defaults():
    s = Step(tool="get_logs")
    assert s.params_contain is None
    assert s.optional is False
    assert s.must_follow is None


def test_step_custom_fields():
    s = Step(tool="run_playbook", must_follow="get_logs", optional=False)
    assert s.must_follow == "get_logs"
    assert s.optional is False


def test_toolcallgraph_from_expected():
    eg = ExpectedGraph(
        steps=[
            DatasetStep(tool="get_logs", params_contain={"service": "payment"}),
            DatasetStep(tool="run_playbook", must_follow="get_logs"),
            DatasetStep(tool="notify_oncall", optional=True),
        ],
        max_steps=6,
        must_terminate=True,
    )
    graph = ToolCallGraph.from_expected(eg)
    assert len(graph.steps) == 3
    assert graph.max_steps == 6
    assert graph.must_terminate is True
    assert graph.steps[0].tool == "get_logs"
    assert graph.steps[0].params_contain == {"service": "payment"}
    assert graph.steps[1].must_follow == "get_logs"
    assert graph.steps[2].optional is True


def test_toolcallgraph_required_steps():
    graph = ToolCallGraph(
        steps=[
            Step(tool="get_logs"),
            Step(tool="run_playbook"),
            Step(tool="notify_oncall", optional=True),
        ]
    )
    required = graph.required_steps
    assert len(required) == 2
    assert all(not s.optional for s in required)


def test_toolcallgraph_step_names():
    graph = ToolCallGraph(
        steps=[
            Step(tool="get_logs"),
            Step(tool="run_playbook"),
            Step(tool="notify_oncall"),
        ]
    )
    assert graph.step_names == ["get_logs", "run_playbook", "notify_oncall"]


def test_toolcallgraph_defaults():
    graph = ToolCallGraph(steps=[])
    assert graph.max_steps == 10
    assert graph.must_terminate is True
