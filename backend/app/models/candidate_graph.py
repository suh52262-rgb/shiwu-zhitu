"""候选 Problem Graph：Auto Generator 的输出协议（P14）。

与 `ProblemGraph` 的区别（任务书 P14 的输出建议）：

- 多出 `knowledge_points`（整题层面的知识点汇总）；
- 多出 `alternative_methods`（"这道题还有哪些合理路线"，供教师复核与后续 Step Matcher 接受替代解）；
- 带 `generation` 来源元信息。

**它只是候选，不默认视为绝对正确**（任务书 P14 明确要求）。
因此转换出的 `ProblemGraph.metadata` 会带 `source="auto"` 与 `needs_review=True`，
深度校验（结构 / 流程完整性 / 物理一致性 / 多路径一致性）留给 P15 Graph Validator。
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.models.problem_graph import GraphNode, ProblemGraph

# 生成来源标记，写入 ProblemGraph.metadata
SOURCE_AUTO = "auto"


class GenerationInfo(BaseModel):
    """这份候选图是怎么来的（便于教师复核与问题回溯）。"""

    source: str = SOURCE_AUTO          # auto / human / teacher
    model: str = ""                    # 生成时使用的模型名
    provider: str = ""                 # 生成时使用的 provider
    course: str = "大学物理"
    generated_at: str = ""             # ISO 时间戳
    needs_review: bool = True          # 候选图默认需要人工/校验器确认
    normalizations: list[str] = Field(default_factory=list)   # 生成后做了哪些修正
    notes: str = ""


class CandidateGraph(BaseModel):
    """LLM 生成的候选解题图（任务书 P14 的输出结构）。"""

    problem_id: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    valid_paths: list[list[str]] = Field(default_factory=list)
    alternative_methods: list[str] = Field(default_factory=list)
    generation: GenerationInfo = Field(default_factory=GenerationInfo)
    node_evidence: dict[str,list[str]] = Field(default_factory=dict)
    quantity_coverage: dict[str,list[str]] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)

    # ---------- 辅助方法 ----------

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def terminal_nodes(self) -> set[str]:
        return self.node_ids() - {src for src, _ in self.edges}

    def to_problem_graph(self) -> ProblemGraph:
        """转成 Agent 真正消费的 ProblemGraph，并把来源写进 metadata。"""
        metadata = {
            "source": self.generation.source,
            "needs_review": self.generation.needs_review,
            "generated_at": self.generation.generated_at,
            "model": self.generation.model,
            "provider": self.generation.provider,
            "course": self.generation.course,
            "knowledge_points": list(self.knowledge_points),
            "alternative_methods": list(self.alternative_methods),
            "normalizations": list(self.generation.normalizations),
            "node_evidence": self.node_evidence,
            "quantity_coverage": self.quantity_coverage,
            "open_questions": self.open_questions,
        }
        if self.generation.notes:
            metadata["notes"] = self.generation.notes
        return ProblemGraph(
            problem_id=self.problem_id,
            nodes=list(self.nodes),
            edges=[list(edge) for edge in self.edges],
            valid_paths=[list(path) for path in self.valid_paths],
            metadata=metadata,
        )


def new_generation_info(
    *,
    model: str = "",
    provider: str = "",
    course: str = "大学物理",
    notes: str = "",
) -> GenerationInfo:
    return GenerationInfo(
        model=model,
        provider=provider,
        course=course,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
