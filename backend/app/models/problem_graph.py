"""Problem-Solving Graph（统一数据协议，任务书第 4.2 节）。

描述一道题可能的合理解题过程：节点 = 解题步骤，边 = 推进关系，
valid_paths = 允许的完整路径（可有多条等价路径）。

Graph 是独立数据资源（`data/problems/{problem_id}/graph.json`，未来可入库），
允许人工建立 / AI 生成 / 教师修改——Agent 不依赖其具体来源，只依赖本 Schema。
"""
from pydantic import BaseModel, Field

# 建议的节点类型（不强制枚举，便于知识工程侧扩展）
NODE_TYPES = ("modeling", "key_reasoning", "calculation", "verification", "result")


class GraphNode(BaseModel):
    """解题步骤节点。"""

    id: str                                  # 如 "S3"
    name: str                                # 如 "确定分离条件"
    type: str = "modeling"                   # 见 NODE_TYPES
    description: str = ""                    # 该步骤的物理内容（供 Step Matcher 判断）
    knowledge_points: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)   # 典型错误描述（E1/E2…或自然语言）


class ProblemGraph(BaseModel):
    problem_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)      # ["S1","S2"] 表示 S1→S2
    valid_paths: list[list[str]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    # ---------- 辅助方法 ----------

    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def successors(self, node_id: str) -> list[str]:
        return [dst for src, dst in self.edges if src == node_id]

    def terminal_nodes(self) -> set[str]:
        """终端节点：没有后继的节点（学生走完即视为完成整题）。"""
        return self.node_ids() - {src for src, _ in self.edges}

    def render_for_prompt(self) -> str:
        """把 Graph 序列化为给 LLM（Step Matcher）看的紧凑文本。"""
        lines: list[str] = []
        for node in self.nodes:
            parts = [f"{node.id} {node.name}（{node.type}）"]
            if node.description:
                parts.append(f"内容：{node.description}")
            if node.knowledge_points:
                parts.append("知识点：" + "、".join(node.knowledge_points))
            if node.common_errors:
                parts.append("常见错误：" + "；".join(node.common_errors))
            lines.append("- " + "｜".join(parts))
        if self.edges:
            lines.append(
                "推进关系：" + "，".join(f"{src}→{dst}" for src, dst in self.edges)
            )
        if self.valid_paths:
            for i, path in enumerate(self.valid_paths, 1):
                lines.append(f"完整路径{i}：{'→'.join(path)}")
        return "\n".join(lines)
