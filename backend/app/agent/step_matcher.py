"""Step Matcher（P2 核心模块，任务书第 5 节）。

目标：把学生的自然语言 / 公式作答映射到 Problem Graph 的节点。

需要支持（任务书要求）：
- 一个输入匹配一个节点；一个输入覆盖多个节点（跳步）；
- 合理替代路径（如能量法 / 简谐运动法）；
- 结论正确但关键推理缺失（correct=True 且 missing_reasoning 非空）；
- 无法匹配任何节点（off_graph=True）；
- 错误作答（matched 为空，但 attempted_nodes 指出学生在哪个节点上出错）。

输入包含题目、Graph、已访问节点与近期事件，输出为结构化 StepMatch。
只依赖 ProblemGraph Schema，不依赖 Graph 的来源（人工 / AI / 教师）。
"""
from pydantic import BaseModel, Field

from app.agent.prompt_registry import get_prompt
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState, TimelineEvent
from app.services.llm import LLMClient, get_llm
from app.services.subquestion_context import context_block


class MatchOutcome(BaseModel):
    """Step Matcher 的弹性结果（P20）。

    `degraded=True` 表示这次匹配来自降级链（模型不可用 / 输出解析失败 /
    只抢救出部分字段）。**调用方不得把降级结果当成真实匹配写进学生状态**：
    空匹配的语义是"学生什么都没做对"，采用它会冤枉学生。
    """

    match: "StepMatch"
    degraded: bool = False
    reasons: list[str] = Field(default_factory=list)


class StepMatch(BaseModel):
    """Step Matcher 结构化输出。"""

    matched_nodes: list[str] = Field(
        default_factory=list,
        description="实质正确完成的节点 id（结论或推理错误的不算完成）",
    )
    attempted_nodes: list[str] = Field(
        default_factory=list,
        description="学生正在尝试但未正确完成的节点 id（错误作答时填写）",
    )
    confidence: float = 0.0
    correct: bool = False
    current_method: str = ""
    missing_reasoning: list[str] = Field(default_factory=list)
    off_graph: bool = Field(
        default=False, description="输入与解题完全无关（闲聊等），与图上任何节点都不相关"
    )
    explanation: str = Field(default="", description="一句话判断依据（供 trace 与教师查看）")


def _recent_events_text(timeline: list[TimelineEvent], limit: int = 5) -> str:
    if not timeline:
        return "无"
    tail = timeline[-limit:]
    return "；".join(
        f"t={event.timestamp:.0f}s {event.event_type}: {event.content[:50]}"
        for event in tail
    )


def match_step(
    problem: Problem,
    graph: ProblemGraph,
    state: StudentState,
    student_input: str,
    timeline: list[TimelineEvent] | None = None,
    llm: LLMClient | None = None,
) -> StepMatch:
    """把学生当前输入映射到 Graph 节点。

    图中不存在的节点 id 会被后校验过滤（防止 LLM 编造节点）；
    matched 与 attempted 均为空时才视为 off_graph。
    """
    llm = llm or get_llm("step_matcher")
    user = (
        f"【题目】\n{problem.statement}\n\n"
        f"【Problem Graph】\n{graph.render_for_prompt()}\n\n"
        f"【学生已完成的节点】{('、'.join(state.visited_nodes)) or '无'}\n\n"
        f"【近期学习事件】{_recent_events_text(timeline or [])}\n\n"
        f"【学生当前输入】\n{student_input.strip() or '（空）'}" + context_block(state.subquestion_context)
    )
    result = llm.chat_structured(get_prompt("step_matcher"), user, StepMatch)

    # ---- 后校验：过滤不存在的节点 id；全部无效时清空并判定为 off_graph ----
    # 注意：这里**返回校验后的副本**，不就地改写 result。
    # 曾经的写法是直接改 result 的字段，一旦某个 LLM 实现复用了同一个对象
    # （测试夹具就是如此），就会把那份对象永久改坏，后续调用全部拿到空匹配。
    valid_ids = graph.node_ids()
    matched = [nid for nid in result.matched_nodes if nid in valid_ids]
    attempted = [
        nid for nid in result.attempted_nodes if nid in valid_ids and nid not in matched
    ]
    return result.model_copy(
        update={
            "matched_nodes": matched,
            "attempted_nodes": attempted,
            "off_graph": result.off_graph or (not matched and not attempted),
            "confidence": min(max(result.confidence, 0.0), 1.0),
        }
    )


def match_step_resilient(
    problem,
    graph,
    state,
    student_input: str,
    timeline: list[TimelineEvent] | None = None,
    llm: LLMClient | None = None,
) -> MatchOutcome:
    """`match_step` 的弹性版本（P20）：模型不可用或输出解析失败时**不抛异常**。

    与严格版的区别只在于失败处理——评测脚本需要看见真实失败，所以 `match_step`
    保持原样不抛不吞；图的工作流则必须活得下来。

    降级时返回一个"空匹配"，但**调用方必须看 `degraded` 并拒绝采用它**：
    空匹配的语义是"学生什么都没做对"，那是一个会冤枉学生的判断。
    """
    llm = llm or get_llm("step_matcher")
    user = (
        f"【题目】\n{problem.statement}\n\n"
        f"【Problem Graph】\n{graph.render_for_prompt()}\n\n"
        f"【学生已完成的节点】{('、'.join(state.visited_nodes)) or '无'}\n\n"
        f"【近期学习事件】{_recent_events_text(timeline or [])}\n\n"
        f"【学生当前输入】\n{student_input.strip() or '（空）'}" + context_block(state.subquestion_context)
    )
    outcome = llm.chat_structured_resilient(get_prompt("step_matcher"), user, StepMatch)
    if outcome.value is None:
        return MatchOutcome(match=_empty_match(), degraded=True, reasons=outcome.reasons)

    match = _post_validate(outcome.value, graph)
    if outcome.degraded:
        # 宽松解析救回来的字段可能是残缺的（例如 matched 捞到了、correct 没捞到）。
        # 这类结果对"学生走到哪一步"仍可能有参考价值，但对"本轮是否正确"不可信，
        # 所以保留匹配、标记降级，由调用方决定用不用。
        return MatchOutcome(match=match, degraded=True, reasons=outcome.reasons)
    return MatchOutcome(match=match)


def _post_validate(result: StepMatch, graph) -> StepMatch:
    """过滤不存在的节点 id；全部无效时清空并判定为 off_graph。"""
    valid_ids = graph.node_ids()
    matched = [nid for nid in result.matched_nodes if nid in valid_ids]
    attempted = [
        nid for nid in result.attempted_nodes if nid in valid_ids and nid not in matched
    ]
    return result.model_copy(
        update={
            "matched_nodes": matched,
            "attempted_nodes": attempted,
            "off_graph": result.off_graph or (not matched and not attempted),
            "confidence": min(max(result.confidence, 0.0), 1.0),
        }
    )


def _empty_match() -> StepMatch:
    return StepMatch(matched_nodes=[], attempted_nodes=[], confidence=0.0, correct=False)
