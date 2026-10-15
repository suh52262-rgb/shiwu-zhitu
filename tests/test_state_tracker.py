"""State Tracker 测试（P3，任务书第 6 节）。"""
from app.agent.state_tracker import apply_step_match
from app.agent.step_matcher import StepMatch
from app.models.student import StudentState


def test_matched_node_enters_visited_and_current():
    state = StudentState(visited_nodes=["S1", "S2"])
    updates = apply_step_match(state, StepMatch(matched_nodes=["S5"], correct=True))

    assert updates["visited_nodes"] == ["S1", "S2", "S5"]  # 保持到达顺序
    assert updates["current_node"] == "S5"
    assert updates["node_attempts"] == {"S5": 1}
    assert updates["node_errors"] == {}


def test_duplicate_match_not_duplicated():
    state = StudentState(visited_nodes=["S5"], node_attempts={"S5": 1})
    updates = apply_step_match(state, StepMatch(matched_nodes=["S5"], correct=True))

    assert updates["visited_nodes"] == ["S5"]
    assert updates["node_attempts"] == {"S5": 2}


def test_attempted_sets_current_but_not_visited():
    """错误作答：定位 current_node，但不计入已访问路径，并记一次节点错误。"""
    state = StudentState(visited_nodes=["S1"])
    updates = apply_step_match(
        state, StepMatch(matched_nodes=[], attempted_nodes=["S10"], correct=False)
    )

    assert updates["visited_nodes"] == ["S1"]
    assert updates["current_node"] == "S10"
    assert updates["node_attempts"] == {"S10": 1}
    assert updates["node_errors"] == {"S10": 1}


def test_wrong_with_matched_counts_error_defensively():
    """LLM 若在 correct=False 时仍返回 matched，防御性处理：不进路径、计错误。"""
    state = StudentState()
    updates = apply_step_match(state, StepMatch(matched_nodes=["S5"], correct=False))

    assert updates["visited_nodes"] == []
    assert updates["current_node"] == "S5"
    assert updates["node_errors"] == {"S5": 1}


def test_method_updated_only_when_present():
    updates = apply_step_match(StudentState(), StepMatch(matched_nodes=["S1"], correct=True))
    assert "current_method" not in updates

    updates = apply_step_match(
        StudentState(), StepMatch(matched_nodes=["S6a"], correct=True, current_method="能量法")
    )
    assert updates["current_method"] == "能量法"


def test_missing_reasoning_recorded():
    updates = apply_step_match(
        StudentState(),
        StepMatch(matched_nodes=["S5"], correct=True, missing_reasoning=["缺少 N=0 判断"]),
    )
    assert updates["missing_reasoning"] == ["缺少 N=0 判断"]


def test_node_errors_accumulate_across_rounds():
    state = StudentState(node_errors={"S10": 1}, node_attempts={"S10": 1})
    updates = apply_step_match(
        state, StepMatch(matched_nodes=[], attempted_nodes=["S10"], correct=False)
    )
    assert updates["node_errors"] == {"S10": 2}
    assert updates["node_attempts"] == {"S10": 2}


def test_off_graph_input_changes_nothing():
    updates = apply_step_match(
        StudentState(visited_nodes=["S1"]),
        StepMatch(matched_nodes=[], attempted_nodes=[], correct=False, off_graph=True),
    )
    assert updates["visited_nodes"] == ["S1"]
    assert "current_node" not in updates
    assert updates["node_attempts"] == {}
    assert updates["node_errors"] == {}
    assert "consecutive_failures" not in updates  # 无关输入不改变计数


# ---- 连续失败计数（P5 Hint Policy 输入） ----

def test_correct_progress_resets_failure_count():
    state = StudentState(consecutive_failures=3)
    updates = apply_step_match(state, StepMatch(matched_nodes=["S7"], correct=True))
    assert updates["consecutive_failures"] == 0


def test_failed_attempt_increments_failure_count():
    state = StudentState(consecutive_failures=2)
    updates = apply_step_match(
        state, StepMatch(matched_nodes=[], attempted_nodes=["S10"], correct=False)
    )
    assert updates["consecutive_failures"] == 3


def test_no_update_key_when_neither_success_nor_failure():
    state = StudentState()
    updates = apply_step_match(state, StepMatch(matched_nodes=[], correct=False, off_graph=True))
    assert "consecutive_failures" not in updates
