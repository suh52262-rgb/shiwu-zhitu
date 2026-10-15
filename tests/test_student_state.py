"""StudentState 核心模型测试（用开发文档第三节示例校验）。"""
from app.models.student import StudentState


def test_student_state_from_doc_example():
    state = StudentState(
        student_id="stu_001",
        problem_id="phy_001",
        current_step="判断机械能是否守恒",
        diagnosed_issue="未考虑摩擦力做功",
        weak_concepts=["机械能守恒条件"],
        stuck_score=0.76,
        hint_level=1,
        finished=False,
    )
    assert state.weak_concepts == ["机械能守恒条件"]
    assert 0 <= state.stuck_score <= 1
    assert state.hint_level == 1
    assert state.finished is False


def test_student_state_defaults_allow_empty_init():
    """LangGraph invoke({}) 时依赖这些默认值。"""
    state = StudentState()
    assert state.student_id == ""
    assert state.timeline == []
    assert state.retrieved_context == []
    assert state.stuck_score == 0.0
