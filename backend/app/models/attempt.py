"""作答记录模型：Attempt 与 StudentState 快照（P11）。

任务书《P11-P25 平台化任务书》（docs/specs/03-第三阶段-P11-P25-平台化.md）第 P11 节把"学生做一道题的全过程"抽象为
Attempt = 某学生 + 某道题 + 一次完整作答过程，生命周期：

    开始作答 → 多轮输入 → 多次诊断 → 多次提示 → 完成 / 放弃 → 保存最终学习状态

关键约束（任务书 P11）：LangGraph 每次运行前 load StudentState、运行完成后
save StudentState，**不要依赖 LLM 自己记住之前发生了什么**；服务重启后再加载
attempt，`current_node` / `visited_nodes` / `mistakes` / `hint_history` 必须仍在。

字段命名遵循任务书（`started_at` / `updated_at` / `finished_at`）；存储层沿用
Phase 6 已有的列名 `start_time` / `end_time`（见 `db/tables.py`），映射集中在
`services/attempt_store.py`，以免改动既有查询与 API 响应。

一次 Attempt 同时保存两份表示，由 attempt_store 在同一处写入以防漂移：

- **状态投影**（`current_node` / `visited_nodes` / `mistakes` / `hint_history` …）：
  可直接查询与聚合，供前端展示与后续 P24 Teacher API 统计；
- **`state`**：完整 `StudentState` 快照，用于精确恢复 Agent 工作状态。
"""
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.student import StudentState

# Attempt 生命周期状态
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINISHED = "finished"
STATUS_ABANDONED = "abandoned"
ATTEMPT_STATUSES = (STATUS_IN_PROGRESS, STATUS_FINISHED, STATUS_ABANDONED)


class AttemptRecord(BaseModel):
    """一次作答过程的完整记录（对应 attempts 表一行 + state 快照）。"""

    attempt_id: str = ""
    student_id: str = ""
    problem_id: str = ""
    status: str = STATUS_IN_PROGRESS

    # ---- 状态投影（任务书 P11 要求 Attempt 直接持有的字段）----
    current_node: str = ""                    # 当前所在 Problem Graph 节点
    current_method: str = ""                  # 当前解题方法（如"能量法"）
    visited_nodes: list[str] = Field(default_factory=list)
    mistakes: list[dict] = Field(default_factory=list)          # 逐轮诊断出的错误
    missing_reasoning: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    stuck_score: float = 0.0
    hint_history: list[dict] = Field(default_factory=list)      # 逐条提示摘要

    # ---- 计数：卡顿检测与提示策略的输入，跨请求必须持久 ----
    error_count: int = 0
    hint_count: int = 0
    stuck_count: int = 0

    # ---- 时间 ----
    started_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None       # 存储层列名为 end_time
    final_correct: bool | None = None

    # ---- 完整 Agent 状态快照（精确恢复用）----
    state: StudentState | None = None

    @property
    def is_open(self) -> bool:
        """是否仍在作答中（可被续做）。"""
        return self.status == STATUS_IN_PROGRESS


# 兼容旧名字：Phase 6 起 models 包对外导出的是 Attempt
Attempt = AttemptRecord
