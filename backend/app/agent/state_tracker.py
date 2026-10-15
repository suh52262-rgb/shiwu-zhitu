"""State Tracker（P3，任务书第 6 节）。

核心原则：系统必须持续知道学生真正走过了哪些节点、在哪个节点上反复出错，
而不是每轮让 LLM 从头猜测。

实现为纯函数：输入当前 StudentState 与 StepMatch，
返回需要合并进状态的字段子集（dict）——与 LangGraph 节点的更新约定一致，
API 层与未来的图节点都可以直接使用。
"""
from app.agent.step_matcher import StepMatch
from app.models.student import StudentState


def apply_step_match(state: StudentState, match: StepMatch) -> dict:
    """把一轮 StepMatch 结果合并进学生状态，返回字段更新 dict。

    规则：
    - visited_nodes 只收录"正确完成"的节点（matched_nodes 且 correct=True），
      保持首次到达顺序、不重复；
    - current_node：刚完成的节点 > 正在尝试的节点 > 保持不变；
    - node_attempts：matched + attempted 都计一次尝试；
    - node_errors：仅当本轮结论/推理不正确时，在尝试节点（退化时用 matched）上计错误；
    - missing_reasoning：记录本轮缺失的关键推理（覆盖上一轮）。
    """
    updates: dict = {}

    completed = match.matched_nodes if match.correct else []

    merged = list(state.visited_nodes)
    for node_id in completed:
        if node_id not in merged:
            merged.append(node_id)
    updates["visited_nodes"] = merged

    if match.matched_nodes:
        updates["current_node"] = match.matched_nodes[-1]
    elif match.attempted_nodes:
        updates["current_node"] = match.attempted_nodes[-1]

    if match.current_method:
        updates["current_method"] = match.current_method

    attempts = dict(state.node_attempts)
    for node_id in dict.fromkeys(match.matched_nodes + match.attempted_nodes):
        attempts[node_id] = attempts.get(node_id, 0) + 1
    updates["node_attempts"] = attempts

    errors = dict(state.node_errors)
    if not match.correct:
        error_nodes = match.attempted_nodes or match.matched_nodes
        for node_id in dict.fromkeys(error_nodes):
            errors[node_id] = errors.get(node_id, 0) + 1
    updates["node_errors"] = errors

    # 无关输入不该抹掉上一轮指出的缺失推理——学生随口问一句，
    # 下一轮 Tutor 的【步骤定位】块就再也拿不到"还缺哪几步推理"了。
    if not match.off_graph:
        updates["missing_reasoning"] = list(match.missing_reasoning)

    # 连续失败计数（Hint Policy 输入，P5）：推进成功归零，失败累加，无关输入不变
    if match.off_graph:
        pass
    elif completed:
        updates["consecutive_failures"] = 0
    else:
        updates["consecutive_failures"] = state.consecutive_failures + 1

    return updates
