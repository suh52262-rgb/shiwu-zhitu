"""卡顿检测器。

P3（第一版）：规则型，基于行为信号（idle / error / rewrite / hint）计算
stuck_score ∈ [0, 1]，超过阈值交给 Agent 进一步分析。

P4（Graph-aware 增强层）：在行为信号之上叠加 Problem Graph 信号——
关键节点、同一节点反复尝试、节点反复出错、相关知识点历史薄弱。
分数仍为规则计算；卡顿原因解释由 LLM（diagnose / hint 节点）负责，
本模块只提供可读的规则归因列表辅助 trace 与前端展示。

规则参数集中定义，未来可整体替换为更复杂的模型，接口保持不变。
"""
from dataclasses import dataclass

from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState


@dataclass
class StuckSignals:
    """卡顿检测输入信号（由 Learning Timeline 统计得出）。"""

    idle_time: float = 0.0     # 距上次有效操作的空闲秒数
    error_count: int = 0       # 明显错误次数
    rewrite_count: int = 0     # 改写 / 擦除重写次数
    hint_count: int = 0        # 已请求提示次数


# ---- 行为规则参数（集中定义，便于调参）----
WEIGHTS = {
    "idle": 0.30,
    "error": 0.25,
    "rewrite": 0.25,
    "hint": 0.20,
}
IDLE_TIME_THRESHOLD = 40.0   # 秒
ERROR_COUNT_THRESHOLD = 2
REWRITE_COUNT_THRESHOLD = 3
HINT_COUNT_THRESHOLD = 1

# stuck_score 超过该值视为疑似卡顿，交 Agent 进一步分析
STUCK_THRESHOLD = 0.50


# ---- 图感知信号（P4）----

@dataclass
class GraphSignals:
    """Problem Graph 感知的卡顿信号。"""

    current_node: str = ""               # 当前所在节点 id（空 = graphless）
    current_node_type: str = ""          # 节点类型（key_reasoning 等）
    is_key_node: bool = False            # 关键节点（key_reasoning 或 metadata.key_nodes 标注）
    node_retry_count: int = 0            # 当前节点累计尝试次数
    node_error_count: int = 0            # 当前节点累计错误次数
    weak_concept_overlap: bool = False   # 节点知识点与学生薄弱知识点重叠


GRAPH_WEIGHTS = {
    "key_node_stuck": 0.10,      # 在关键节点上反复尝试（>=2 次）
    "same_node_repeat": 0.10,    # 同一节点尝试 >= 3 次
    "node_error_repeat": 0.15,   # 当前节点错误 >= 2 次
    "weak_concept": 0.10,        # 相关知识点历史表现差
}
KEY_NODE_RETRY_THRESHOLD = 2
SAME_NODE_RETRY_THRESHOLD = 3
NODE_ERROR_THRESHOLD = 2


def compute_stuck_score(signals: StuckSignals) -> float:
    """按行为规则累加分数，结果限制在 0~1。"""
    score = 0.0
    if signals.idle_time > IDLE_TIME_THRESHOLD:
        score += WEIGHTS["idle"]
    if signals.error_count >= ERROR_COUNT_THRESHOLD:
        score += WEIGHTS["error"]
    if signals.rewrite_count >= REWRITE_COUNT_THRESHOLD:
        score += WEIGHTS["rewrite"]
    if signals.hint_count >= HINT_COUNT_THRESHOLD:
        score += WEIGHTS["hint"]
    return min(score, 1.0)


def build_graph_signals(graph: ProblemGraph | None, state: StudentState) -> GraphSignals:
    """从 ProblemGraph 与 StudentState 提取图感知信号。"""
    if graph is None or not state.current_node:
        return GraphSignals()
    node = graph.node_map().get(state.current_node)
    if node is None:
        return GraphSignals()
    key_nodes = set(graph.metadata.get("key_nodes", []))
    return GraphSignals(
        current_node=node.id,
        current_node_type=node.type,
        is_key_node=node.type == "key_reasoning" or node.id in key_nodes,
        node_retry_count=state.node_attempts.get(node.id, 0),
        node_error_count=state.node_errors.get(node.id, 0),
        weak_concept_overlap=bool(
            set(e.get("concept") for e in state.error_evidence
                if e.get("issue") and e.get("student_text")) & set(node.knowledge_points)
        ),
    )


def compute_graph_aware_stuck_score(signals: StuckSignals, graph: GraphSignals) -> float:
    """行为规则分 + 图感知加成，结果限制在 0~1。

    图信号让系统在行为分数不高时也能提前感知"卡在关键节点上"，
    例如：在 S3（分离条件）上连续两次推理错误，即使没有空闲也会加分。
    """
    score = compute_stuck_score(signals)
    if graph.is_key_node and graph.node_retry_count >= KEY_NODE_RETRY_THRESHOLD:
        score += GRAPH_WEIGHTS["key_node_stuck"]
    if graph.node_retry_count >= SAME_NODE_RETRY_THRESHOLD:
        score += GRAPH_WEIGHTS["same_node_repeat"]
    if graph.node_error_count >= NODE_ERROR_THRESHOLD:
        score += GRAPH_WEIGHTS["node_error_repeat"]
    if graph.weak_concept_overlap:
        score += GRAPH_WEIGHTS["weak_concept"]
    return min(score, 1.0)


def stuck_reasons(signals: StuckSignals, graph: GraphSignals) -> list[str]:
    """返回触发的卡顿信号的可读解释（规则归因，供 trace / 前端展示）。"""
    reasons: list[str] = []
    if signals.idle_time > IDLE_TIME_THRESHOLD:
        reasons.append(f"空闲超过 {IDLE_TIME_THRESHOLD:.0f} 秒")
    if signals.error_count >= ERROR_COUNT_THRESHOLD:
        reasons.append(f"累计明显错误 {signals.error_count} 次")
    if signals.rewrite_count >= REWRITE_COUNT_THRESHOLD:
        reasons.append(f"反复改写 {signals.rewrite_count} 次")
    if signals.hint_count >= HINT_COUNT_THRESHOLD:
        reasons.append(f"已请求 {signals.hint_count} 次提示")
    if graph.is_key_node and graph.node_retry_count >= KEY_NODE_RETRY_THRESHOLD:
        reasons.append(
            f"在关键节点 {graph.current_node}（{graph.current_node_type}）上已尝试 "
            f"{graph.node_retry_count} 次"
        )
    if graph.node_retry_count >= SAME_NODE_RETRY_THRESHOLD:
        reasons.append(f"同一节点 {graph.current_node} 已反复尝试 {graph.node_retry_count} 次")
    if graph.node_error_count >= NODE_ERROR_THRESHOLD:
        reasons.append(f"节点 {graph.current_node} 上已出错 {graph.node_error_count} 次")
    if graph.weak_concept_overlap:
        reasons.append(f"节点 {graph.current_node} 涉及历史薄弱知识点")
    return reasons


def is_stuck(stuck_score: float, threshold: float = STUCK_THRESHOLD) -> bool:
    """判断是否疑似卡顿。"""
    return stuck_score >= threshold
