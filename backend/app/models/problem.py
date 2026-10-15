"""题目模型（统一数据协议，任务书第 4.1 节）。

协议字段：problem_id / statement / figures / knowledge_points
扩展字段（教学系统内部使用）：title / answer / difficulty / metadata

兼容说明：旧数据用 content 表示题干，通过 validation_alias 双向兼容——
读入时 content 会映射到 statement，序列化统一输出 statement。
"""
from pydantic import AliasChoices, BaseModel, Field


class Problem(BaseModel):
    problem_id: str
    statement: str = Field(
        default="", validation_alias=AliasChoices("statement", "content")
    )
    figures: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)

    # ---- 扩展字段 ----
    title: str = ""
    answer: str = ""                                   # 参考答案，不下发给前端
    difficulty: int = 1                                # 1~5
    metadata: dict = Field(default_factory=dict)

    @property
    def content(self) -> str:
        """兼容旧代码的题干读取入口。"""
        return self.statement

    @property
    def full_text(self) -> str:
        """下发给 Agent 的完整题面（标题 + 题干）。

        统一出口，避免各处自行拼接导致题面不一致（P11：Attempt 创建与
        状态恢复都要用到同一份题面）。
        """
        return f"{self.title}\n{self.statement}".strip()
