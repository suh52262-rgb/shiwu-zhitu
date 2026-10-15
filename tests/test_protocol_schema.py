"""统一数据协议 Schema 测试（P1，任务书第 4 节）。"""
from app.models.event import LearningEvent
from app.models.problem import Problem
from app.models.student import StudentState


def test_problem_statement_with_content_alias():
    """旧协议字段 content 自动映射到 statement。"""
    problem = Problem(problem_id="p1", content="旧字段题干")
    assert problem.statement == "旧字段题干"
    assert problem.content == "旧字段题干"


def test_problem_statement_canonical_roundtrip():
    problem = Problem(problem_id="p1", statement="新协议题干", figures=["f1.png"])
    assert problem.statement == "新协议题干"
    dumped = problem.model_dump()
    assert dumped["statement"] == "新协议题干"
    assert Problem(**dumped).statement == "新协议题干"


def test_learning_event_matches_protocol():
    """任务书 4.3 的事件字段全部具备（attempt_id 为系统扩展）。"""
    event = LearningEvent(timestamp=87, event_type="modify", content="将 N=mg 修改为 N=mg cosθ")
    assert event.timestamp == 87
    assert event.event_type == "modify"
    assert event.metadata == {}


def test_student_state_graph_fields():
    """任务书 4.4 新增字段：current_node / visited_nodes / current_method。"""
    state = StudentState(
        student_id="stu_001",
        problem_id="phy_002",
        current_node="S3",
        visited_nodes=["S1", "S2"],
        current_method="能量法",
        stuck_score=0.76,
        hint_level=1,
    )
    assert state.current_node == "S3"
    assert state.visited_nodes == ["S1", "S2"]
    assert state.current_method == "能量法"


def test_student_state_graph_fields_default_empty():
    """graphless 模式下这些字段为空即可。"""
    state = StudentState()
    assert state.current_node == ""
    assert state.visited_nodes == []
    assert state.current_method == ""
