"""核心数据模型：学生当前学习状态（StudentState）。

Agent 的核心不是聊天记录，而是学生当前学习状态。
该模型同时作为：
- LangGraph 工作流的 State（见 app/agent/state.py）
- API 层传递对象
- Phase 6 数据库持久化的映射来源

字段设计保持开放，便于后续扩展：长期学习画像、视频事件、教师审核等。
"""
from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    """Learning Timeline 中的单个事件（对应 learning_events 表）。"""

    timestamp: float = 0.0        # 相对 attempt 开始的秒数
    event_type: str = "write"     # start/write/modify/erase/idle/hint_request/submit
    content: str = ""


class StudentState(BaseModel):
    """学生状态：一次作答会话（attempt）内 Agent 感知与决策的全部依据。"""

    # ---- 身份 ----
    student_id: str = ""
    problem_id: str = ""
    # Old snapshots deliberately remain unknown; never infer historical graphs.
    graph_binding_status: str = "unknown"
    graph_version: str = ""
    graph_snapshot: dict = Field(default_factory=dict)
    graph_release_status: str = "unknown"

    # ---- 题目上下文（由 API 层加载后传入，LLM 节点使用）----
    problem_content: str = ""
    problem_version: str = ""
    problem_snapshot: dict = Field(default_factory=dict)
    problem_binding_status: str = "unknown"

    # ---- 当前作答 ----
    media_sources: list[dict] = Field(default_factory=list)  # Original OCR and revision evidence
    current_answer: str = ""
    subquestion_context: dict = Field(default_factory=dict)
    current_step: str = ""                    # analyze_step：学生当前做到哪一步（自然语言）
    current_node: str = ""                    # Step Matcher：Problem Graph 节点 id（graphless 模式为空）
    #: State Tracker：**已正确完成**的节点路径（不是"访问过"的节点）。
    #:
    #: ⚠️ 语义容易混淆，实测踩过：模拟器的 `nodes` 断言用的是
    #: `matched_nodes`（对应上就算，不看对错），`run_eval` 用的是
    #: `matched ∪ attempted`（碰过就算）。**三处三种口径。**
    #: 详见 docs/排查-节点轨迹三种口径.md。
    #: 做指标时务必先想清楚要的是"碰过"还是"完成"。
    visited_nodes: list[str] = Field(default_factory=list)
    current_method: str = ""                  # 当前解题方法（如"能量法"）
    concepts: list[str] = Field(default_factory=list)   # analyze_step：涉及知识点
    has_obvious_error: bool | None = None     # analyze_step：是否存在明显错误
    is_complete: bool = False                 # analyze_step：是否已完成整道题
    rag_materials_in_prompt: bool = False     # 本轮Tutor请求正文含教材，不代表语义或网络送达验证
    hint_optional: bool = False
    hint_evidence: list[dict] = Field(default_factory=list)
    diagnosis_text_kind: str = 'legacy'
    #: P23：建议学生下一步做什么（analyze_step）。无 Graph 时这是唯一的"下一步"指引
    recommended_next_step: str = ""
    timeline: list[TimelineEvent] = Field(default_factory=list)

    # ---- 诊断结果（diagnose 节点输出）----
    diagnosed_issue: str = ""                 # 如："未考虑摩擦力做功"
    issue_type: str = ""                      # concept/modeling/method/calculation/verification/none
    severity: str = ""                        # low/medium/high
    recommended_action: str = ""
    weak_concepts: list[str] = Field(default_factory=list)
    profile_policy_version: str = "legacy_unknown"
    legacy_weak_concepts: list[str] = Field(default_factory=list)
    error_evidence: list[dict] = Field(default_factory=list)
    help_events: list[dict] = Field(default_factory=list)
    stuck_score: float = 0.0                  # 0~1，规则型 stuck_detector 输出
    hint_level: int = 0                       # 0=未提示；越高提示越具体

    # ---- 辅导输出 ----
    hint: str = ""                            # 最近一次给学生的提示

    # ---- RAG 检索上下文（Phase 4/5 填充）----
    retrieved_context: list[dict] = Field(default_factory=list)
    rag_query: str = ""                       # 本轮实际使用的 RAG 检索语句（Tracing 用，P9）
    #: ⚠️ **RAG 运行状态**（2026-10-10 核查第 2 项）：**不能把故障显示成"教材里没有"**。
    #: ```
    #: {"state": "not_triggered"|"no_hit"|"failed"|"ok",
    #:  "reason": "…（failed 时给）", "returned": 3,
    #:  "rerank": "ok"|"fallback"|"timeout"|"failed"|"unknown",
    #:  "timings": {"embed_sec": …, "search_sec": …, "rerank_sec": …, "total_sec": …}}
    #: ```
    #: ⚠️ **与"审计状态"是两套**：审计那套判的是"重放与录制是否逐字一致"，
    #: 这套判的是"**本轮到底有没有用上教材、卡在哪一步**"。**别混用。**
    rag_status: dict = Field(default_factory=dict)

    # ---- State Tracker（P3：图感知的过程记录）----
    missing_reasoning: list[str] = Field(default_factory=list)  # 最近一轮缺失的关键推理
    node_attempts: dict[str, int] = Field(default_factory=dict)  # 各节点被尝试次数（含错误尝试）
    node_errors: dict[str, int] = Field(default_factory=dict)    # 各节点上结论/推理错误的次数
    consecutive_failures: int = 0             # 连续未推进轮数（Hint Policy 输入，P5）
    last_hint_node: str = ""                  # 最近一次下发提示时所在的节点（重复提示检测）
    #: 距上次有效操作秒数。由 API 层从请求带入。
    #:
    #: P21：干预控制器用它区分"学生在想"与"学生卡住了"——
    #: **单纯发呆不触发提示**，只在结合失败时才作为加压依据。
    idle_time: float = 0.0
    #: 已下发的提示记录，按时间顺序，每项 `{"node": ..., "level": ...}`。
    #:
    #: P21：用来回答"**这个节点上已经提示过几次**"。比 `last_hint_node`
    #: 一个字符串细得多——知道提示过 2 次还是 1 次，对是否升级的判断不一样。
    hint_history: list[dict] = Field(default_factory=list)
    #: P21：本轮的干预动作（observe / withhold / nudge / guide / explain）。
    #: **与 `hint_level` 一起看**：`level=0` 可能有两种完全不同的含义——
    #: "没事发生"（observe）与"本来能给、故意不给"（withhold）。
    #: 前端与 Trace 靠这个字段区分，不必反推。
    hint_action: str = "observe"
    #: P21：抑制时的理由（`hint_action == "withhold"` 时非空）
    hint_withheld_reason: str = ""

    # ---- 单轮分析标记（P6 条件路由使用，每轮由节点覆盖）----
    graph_mode: bool = False                  # 本题是否有 Problem Graph（API 层设置）
    request_hint: bool = False                # 学生本轮是否主动请求提示
    round_correct: bool = False               # 本轮是否正确推进（StepMatch 产出）
    round_off_graph: bool = False             # 本轮输入是否与题目无关
    hint_reasons: list[str] = Field(default_factory=list)      # Hint Policy 决策依据
    last_step_match: dict = Field(default_factory=dict)        # 最近一次 StepMatch 完整结果（响应/Tracing）
    #: P20：本轮降级原因（每轮覆盖）。非空表示"本轮有环节没能正常工作"，
    #: 例如模型输出解析不了、调用超时。**降级不等于出错**——工作流仍会返回，
    #: 但调用方（前端/教师端）应当知道这一轮的判断可能不完整。
    degraded_reasons: list[str] = Field(default_factory=list)

    #: ⚠️ **安全替换**（2026-10-10）：某段**文本**（目前只有 hint）被服务端
    #: **换成了本地保守文本**（不调用模型、不冒充模型输出）。
    #:
    #: 🔴 **与 `degraded_reasons` 是两种状态，绝不许混用**（用户裁决）：
    #: ```
    #: · degraded_reasons     = **诊断 / 判分 / 输入依据不可靠** ⇒ 影响 reliable、finished、案例可评性
    #: · safety_substitutions = **判断没问题，只是某段文本被安全替换** ⇒ **不改变**上面三者
    #: ⇒ 两者同时发生时**两类都记**；安全替换**不得清除**真实降级。
    #: ```
    #: ⚠️ **"判断没问题"必须有独立依据**：若替换的触发原因**同时暴露了批改里的无依据判断**
    #:   （`diagnosed_issue` 也命中同一判据），那就**不能只当提示替换** ⇒ 照记 `degraded_reasons`。
    #:
    #: 元素形如：`{"field": "hint", "detector": "unsupported_claims",
    #:           "reasons": ["你用的"], "original": "<模型原文>", "prompt_version": "v1"}`
    #: `original` 是**审计用**的原模型提示 —— 它**不会**给学生（学生只看到替换后的保守提示）。
    safety_substitutions: list[dict] = Field(default_factory=list)

    # ---- 过程标记 ----
    finished: bool = False
