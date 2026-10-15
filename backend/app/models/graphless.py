"""Graphless Mode 的结构化输出（P23，任务书第三阶段 P23 节）。

任务书要求：即使没有 Problem Graph，也要能基本判断"当前大致处于什么阶段 /
是否明显错误 / 哪个知识点相关 / 是否应该提示"，并**至少输出**：

    {"current_stage", "method", "correct", "issue_type",
     "knowledge_points", "missing_reasoning", "recommended_next_step"}

**为什么单独一个模型，而不是复用 AnalyzeResponse**：
graph 模式的判断是**节点锚定**的（学生走到 S5、S5 是关键节点），
graphless 只能给**阶段级**判断。把两者塞进同一组字段会让人误以为
graphless 的 `current_stage` 也能当节点用。分开表达，边界才清楚。

`correct` 是**三态**：`None` 表示"本轮无法判定"（例如作答为空、或模型没给出结论），
这与 `False`（明确判断为错）是不同的事——下游不该把"不知道"当成"做错了"。
"""
from pydantic import BaseModel, Field


class GraphlessAssessment(BaseModel):
    """无 Graph 时对本轮作答的结构化判断。"""

    current_stage: str = Field(default="", description="当前大致处于哪个解题阶段（自然语言）")
    method: str = Field(default="", description="学生采用的主要解题方法")
    #: 三态：True 正确 / False 明显错误 / None 本轮无法判定
    correct: bool | None = Field(default=None, description="本轮是否正确推进；None=无法判定")
    issue_type: str | None = Field(default=None, description="错误类型；None=未诊断出错误")
    knowledge_points: list[str] = Field(default_factory=list)
    missing_reasoning: list[str] = Field(default_factory=list)
    recommended_next_step: str = Field(default="", description="建议学生下一步做什么")


def build_assessment(state) -> GraphlessAssessment:
    """从本轮结束后的 StudentState 汇总出 graphless 判断。

    `state` 是 StudentState（或等价的 dict 视图）；只读，不改动任何东西。
    """
    get = state.get if isinstance(state, dict) else lambda key, default=None: getattr(state, key, default)

    has_error = get("has_obvious_error")
    # 三态映射：模型没给出结论时保持 None，不要擅自当作"做对了"
    correct = None if has_error is None else not has_error

    issue_type = get("issue_type") or ""
    # diagnose 短路时会把 issue_type 写成 "none"，那表示"没有错误"而非"某种错误类型"
    normalized_issue = None if issue_type in ("", "none") else issue_type

    return GraphlessAssessment(
        current_stage=get("current_step") or "",
        method=get("current_method") or "",
        correct=correct,
        issue_type=normalized_issue,
        knowledge_points=list(get("concepts") or []),
        missing_reasoning=list(get("missing_reasoning") or []),
        recommended_next_step=get("recommended_next_step") or "",
    )
