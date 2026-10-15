"""SQLAlchemy ORM 表定义（Phase 6）。

对应开发文档第八节的六个实体：
students / problems / attempts / learning_events / student_profiles / hint_records

设计说明：
- students / problems 直接用业务 ID（如 stu_001、phy_001）作主键，便于幂等 upsert
- learning_events.timestamp 为相对 attempt 开始的秒数（Learning Event 统一格式）
- metadata 是 SQLAlchemy 保留属性名，Problem.metadata 落库到 extra 列
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# init_db.py 用来打印确认的表名清单
BASE_TABLES = (
    "students",
    "problems",
    "attempts",
    "learning_events",
    "hint_records",
    "student_profiles",
    "analysis_traces",
)


class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 业务 student_id
    display_name: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Problem(Base):
    __tablename__ = "problems"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 业务 problem_id
    title: Mapped[str] = mapped_column(String(256), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    knowledge_points: Mapped[list | None] = mapped_column(JSON, default=list)
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    answer: Mapped[str] = mapped_column(Text, default="")
    extra: Mapped[dict | None] = mapped_column(JSON, default=dict)  # Problem.metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Attempt(Base):
    """一次完整作答过程（P11 Attempt 生命周期）。

    除 Phase 6 已有的时间/计数列外，P11 增补两组列：

    - **状态投影**（status / current_node / visited_nodes / mistakes / …）：
      可直接查询聚合，供展示与后续 P24 Teacher API 统计；
    - **state**：完整 StudentState 的 JSON 快照，用于精确恢复 Agent 工作状态。

    说明：`end_time` 对应领域模型里的 `finished_at`；改名会牵动既有查询与 API，
    故保留列名，映射集中在 `services/attempt_store.py`。
    """

    __tablename__ = "attempts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # attempt_id (uuid)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    problem_id: Mapped[str] = mapped_column(ForeignKey("problems.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="in_progress", index=True)
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    final_correct: Mapped[bool | None] = mapped_column(Boolean, default=None)
    hint_count: Mapped[int] = mapped_column(Integer, default=0)
    stuck_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)

    # ---- 状态投影（P11）----
    current_node: Mapped[str] = mapped_column(String(32), default="")
    current_method: Mapped[str] = mapped_column(String(64), default="")
    visited_nodes: Mapped[list | None] = mapped_column(JSON, default=list)
    mistakes: Mapped[list | None] = mapped_column(JSON, default=list)
    missing_reasoning: Mapped[list | None] = mapped_column(JSON, default=list)
    weak_concepts: Mapped[list | None] = mapped_column(JSON, default=list)
    stuck_score: Mapped[float] = mapped_column(Float, default=0.0)
    hint_history: Mapped[list | None] = mapped_column(JSON, default=list)

    # ---- 完整学生状态快照（P11：服务重启后可精确恢复）----
    state: Mapped[dict | None] = mapped_column(JSON, default=None)


class LearningEvent(Base):
    """Learning Timeline 的存储层：任何来源的输入最终都变成这里的一行。"""

    __tablename__ = "learning_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[float] = mapped_column(Float, default=0.0)  # 相对开始秒数
    event_type: Mapped[str] = mapped_column(String(32), default="write")
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class HintRecord(Base):
    __tablename__ = "hint_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), index=True
    )
    hint_level: Mapped[int] = mapped_column(Integer, default=1)
    hint: Mapped[str] = mapped_column(Text, default="")
    issue_type: Mapped[str] = mapped_column(String(32), default="")

    # ---- P22：效果追踪 ----
    #: 下发提示时学生所在的节点。判定"有没有推进"必须知道**基准在哪**。
    node: Mapped[str] = mapped_column(String(32), default="")
    #: 提示前的状态快照（visited_nodes 等）。
    #: **没有提示前的状态，就无法判断之后的变化**——这是判定的前提。
    state_before: Mapped[dict | None] = mapped_column(JSON, default=dict)
    #: 效果三态：True=有效 / False=无效 / **NULL=无法判定**。
    #: NULL 不是"无效"——把它当 False 会系统性地低估提示效果。
    effective: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    #: 判定依据（人可读，供复核）
    effective_evidence: Mapped[str] = mapped_column(Text, default="")
    #: 判定时刻。NULL = 还没判定
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class StudentProfile(Base):
    """学生长期画像（第一阶段只积累薄弱知识点，后续扩展）。"""

    __tablename__ = "student_profiles"

    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), primary_key=True)
    weak_concepts: Mapped[list | None] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AnalysisTrace(Base):
    """每轮分析的完整 Tracing（P9 建立，P19 强化，任务书第 14 / P19 节）。

    payload 为 JSON 快照：User Input / Learning Events / StepMatch / Diagnosis /
    Stuck Score / RAG Query + Context / Hint Policy / Tutor Output。
    目的：行为异常时可定位是哪一环出错（匹配 / Graph 数据 / 检索 / 诊断 / 策略 / 话术）。

    P19 把关键维度提升为**独立列**，理由有二：
    1. 能直接查询与聚合，不必在 JSON 里翻找；
    2. P24 教师端要按题目统计卡点，必须有 `problem_id` 这样的可索引列。
    """

    __tablename__ = "analysis_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("attempts.id", ondelete="CASCADE"), index=True
    )
    round_index: Mapped[int] = mapped_column(Integer, default=0)
    # ---- P19 新增的可观测性列 ----
    request_id: Mapped[str] = mapped_column(String(32), default="", index=True)
    problem_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    provider: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(96), default="")
    token_usage: Mapped[dict | None] = mapped_column(JSON, default=dict)
    prompt_versions: Mapped[dict | None] = mapped_column(JSON, default=dict)
    before_state: Mapped[dict | None] = mapped_column(JSON, default=dict)
    payload: Mapped[dict | None] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentImportRun(Base):
    """一次导入的批次记录（审计轨迹 + 索引状态）。

    ## 解决两个具体问题

    1. **源文件版本说不清**：此前源 PDF 的 SHA-256 只打印在控制台上，
       事后无法回答"PG 里这批内容来自哪个版本的 PDF"。
       同名文件内容变了也看不出来。
    2. **索引过期没有标记**：`--no-index` 只写 PG 不写 Qdrant，
       而检索端不知道，会继续把**旧向量**当作最新结果返回。
       导入中途失败时也一样——PG 是新的，Qdrant 是部分的。

    ## 与审计方案的一处差异（如实说明）

    方案建议两张表（批次 + 逐页状态）。这里只做**一张**：
    失败页以列表形式记在 `failed_pages` 里，够用于审计与"哪些页要重跑"。

    没做逐页表，是因为**断点续传已经有别的路**：`--pages` 可只重跑指定页，
    `import_pdf_batch.py` 本来就是分批的。为它单独建一张可能上万行的表，
    收益不抵维护成本。（将来真要做"从第 N 页自动续跑"，再补逐页表。）
    """

    __tablename__ = "document_import_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)   # uuid4
    doc_id: Mapped[str] = mapped_column(String(128), index=True)

    #: 源 PDF 的 SHA-256 —— **区分"同一文件重试"与"同名但内容不同"**
    source_sha256: Mapped[str] = mapped_column(String(64), default="")
    source_name: Mapped[str] = mapped_column(String(256), default="")
    doc_type: Mapped[str] = mapped_column(String(32), default="textbook")
    #: pages（只替换指定页）/ document（整文档替换）/ append
    mode: Mapped[str] = mapped_column(String(24), default="document")
    pages_spec: Mapped[str] = mapped_column(String(128), default="")

    #: 用了哪些模型与提示词版本——事后复现同一批结果需要它
    model: Mapped[str] = mapped_column(String(96), default="")
    prompt_versions: Mapped[dict | None] = mapped_column(JSON, default=dict)

    #: running | ready | index_pending | partial | failed
    #: **检索端只认 ready**，其余一律排除，避免混合新旧结果。
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    pages_total: Mapped[int] = mapped_column(Integer, default=0)
    pages_ok: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[int] = mapped_column(Integer, default=0)
    failed_pages: Mapped[list | None] = mapped_column(JSON, default=list)
    blocks: Mapped[int] = mapped_column(Integer, default=0)
    indexed: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class DocumentBlock(Base):
    """导入文档的一个内容块（教材 / 试卷 → 文字 / 公式 / 插图）。

    ## 为什么要有这张表（而不是只写 Qdrant）

    整本教材导入一次要花约 1000 万 tokens。**只存向量库的话，将来换
    embedding 模型就得重调一遍 VL**（再花一千万）。先把 VL 的产出落在这里，
    重建索引就只是重新 embed 文本，**不用再碰模型**。

    于是分工是：
      · 本表 = **事实来源**（结构化、可审计、可断点续传）
      · Qdrant = **索引**（向量 + 展示用 payload）
      · 文件系统 = 原图（`data/documents/{doc_id}/figures/`）

    ## 字段设计的两个要点

    `text` 与 `embed_text` **必须分开**：前者是原样内容（给人和 LLM 看、
    供渲染），后者才是真正被向量化的串。因为 BGE 是中文文本模型，
    对 `\\frac{g}{2v_0^2\\cos^2\\alpha}` 这种记号的语义表征很差——
    直接拿 LaTeX 去 embed 等于浪费一次向量化。

    `is_generated` 标记**机器生成**的内容（VL 写的图注），
    Tutor 引用时不能把它当成教材原文。
    """

    __tablename__ = "document_blocks"

    #: uuid5(doc_id:page:block_index) —— 幂等重导，重复导入不会产生重复行
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), index=True)
    doc_type: Mapped[str] = mapped_column(String(32), default="textbook")
    page: Mapped[int] = mapped_column(Integer, index=True)
    block_index: Mapped[int] = mapped_column(Integer, default=0)

    #: text | formula | figure_caption | figure
    type: Mapped[str] = mapped_column(String(24), default="text", index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    embed_text: Mapped[str] = mapped_column(Text, default="")
    formula_number: Mapped[str] = mapped_column(String(32), default="")

    bbox: Mapped[list | None] = mapped_column(JSON, default=list)
    figure_path: Mapped[str] = mapped_column(String(256), default="")
    figure_labels: Mapped[list | None] = mapped_column(JSON, default=list)
    section: Mapped[str] = mapped_column(String(128), default="")

    #: 这条内容哪来的：text_layer | caption_anchor | vl —— 排障时一眼看出可信度
    source_kind: Mapped[str] = mapped_column(String(24), default="text_layer")
    #: true = 机器生成（VL 图注），不得冒充教材原文
    is_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    #: 是否参与向量化。**false = 原文保留供审计，但不进检索**。
    #:
    #: 用于隔离扫描件文字层里的坏公式（实测前 200 页 96 行，
    #: 如 `'s = (vo +v)t =( ++at)'`）——它们会被当作"教材原文"返回。
    #: 注意**不要用 is_generated 表达**：那个字段的语义是"模型生成"，
    #: 而 OCR 文字层是扫描件自带的，混为一谈会让"排除机器生成"误伤一大片。
    # server_default 用字符串字面量：类体里 `text` 被同名字段遮蔽了，
    # 写 text("true") 会报 'MappedColumn' object is not callable。
    indexable: Mapped[bool] = mapped_column(Boolean, default=True,
                                           server_default="true")
    #: 不索引的原因（可追溯；静默的过滤器比不过滤更危险）
    note: Mapped[str] = mapped_column(String(128), default="")
    #: 入库时的模型与用量，便于按文档核算成本
    model: Mapped[str] = mapped_column(String(96), default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
