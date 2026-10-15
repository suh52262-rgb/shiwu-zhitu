"""作答与辅导接口（Phase 3/6，P11 改为 Attempt 生命周期驱动）。

POST /attempt/analyze：最小教学闭环的核心入口，调用 LangGraph 工作流。

P11 起 **不再有进程内会话缓存**：每次请求先按 attempt_id（或该学生该题未结束的
attempt）从 AttemptStore 加载 StudentState，跑完工作流再写回。因此服务重启后
状态仍在、Agent 能从原状态继续；PostgreSQL 不可用时 AttemptStore 自动退化为
进程内存储，教学闭环不中断。

提示升级规则：首次作答 hint_level=1；上一轮仍有明显错误则 +1（封顶 3），
提示随之更具体——对应"学生持续卡住时给更具体的帮助"的教学原则。
"""
import logging
import time
import uuid
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, PrivateAttr

from app.agent.graph import tutor_graph
from app.agent.prompt_registry import prompt_version_fields
from app.db import persistence
from app.models.attempt import STATUS_FINISHED, STATUS_IN_PROGRESS, AttemptRecord
from app.models.graphless import GraphlessAssessment, build_assessment
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState, TimelineEvent
from app.observability import get_request_id
from app.services.attempt_store import get_attempt_store
from app.services.llm.tracking import track_usage
from app.services.problem_store import load_problem, load_problem_graph
from app.services.stuck_detector import (
    StuckSignals,
    build_graph_signals,
    compute_graph_aware_stuck_score,
    is_stuck,
    stuck_reasons,
)

router = APIRouter(prefix="/attempt", tags=["attempt"])

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    student_id: str
    problem_id: str
    answer: str
    problem_content: str = ""      # 可选；不传则服务端按 problem_id 从题库加载
    idle_time: float = 0.0         # 距上次有效操作的秒数（前端统计）
    rewrite_count: int = 0         # 本次作答前的改写次数（前端统计）
    request_hint: bool = False     # 学生主动点击"请求提示"
    _media_type: str = PrivateAttr(default="")  # Set only by the media adapter
    _media_timestamp: float = PrivateAttr(default=0.0)
    attempt_id: str = ""           # P11：续做指定 attempt；留空则续做未结束的那次


class AnalyzeResponse(BaseModel):
    """响应字段对应开发文档第九节示例。

    P2 新增 step_match：题目有 Problem Graph 时返回节点匹配结果。
    P11 新增 status：本次作答的生命周期状态（in_progress/finished/abandoned）。
    """

    attempt_id: str
    source_id: str = ""
    source_revision: int = 0
    current_step: str
    current_node: str = ""
    visited_nodes: list[str] = []
    graph_mode: bool = False     # P8：本题是否以 Graph 模式运行
    graph_version: str = ""
    graph_binding_status: str = "unknown"
    graph_release_status: str = "unknown"
    problem_version: str = ""
    problem_binding_status: str = "unknown"
    status: str = STATUS_IN_PROGRESS
    issue: str
    issue_type: str
    severity: str
    stuck_score: float
    weak_concepts: list[str]
    hint_level: int
    hint: str
    hint_optional: bool = False
    hint_evidence: list[dict] = []
    diagnosis_text_kind: str = 'legacy'
    finished: bool
    step_match: dict | None = None
    stuck_reasons: list[str] = []   # P4：本轮触发的卡顿信号（可读归因）
    hint_reasons: list[str] = []    # P5：Hint Policy 决策依据
    #: P23：**仅 graphless 模式**下给出的阶段级判断。
    #: graph 模式的判断是节点锚定的（看 step_match / current_node），
    #: 两者语义不同，因此分开表达而不是混进同一组字段。
    graphless: GraphlessAssessment | None = None
    #: P20：本轮是否有环节降级（模型不可用 / 输出解析失败 / 调用超时）。
    #: `degraded=True` 时本轮**不会给出提示**——因为系统并不知道学生这一步对不对，
    #: 编一句提示比不提示更糟。前端应据此提示"本轮分析暂时不可用"。
    degraded: bool = False
    degraded_reasons: list[str] = []

    #: ⚠️ **安全替换**（2026-10-10）：某段文本（目前只有 `hint`）被服务端换成了
    #: **本地保守文本**。**这与"降级"是两种状态**：
    #: ```
    #: · degraded=True            ⇒ 诊断/判分/输入依据不可靠 ⇒ 判断可能不完整
    #: · safety_substitutions 非空 ⇒ **判断没问题**，只是那段文本不许原样给学生
    #: ⇒ 两者可以同时出现；**替换不掩盖降级**。
    #: ```
    #: 学生看到的是替换后的保守提示；`original` 仅供**审计**（前端不要展示原模型文本）。
    safety_substitutions: list[dict] = []

    #: ⚠️ **本轮 RAG 运行状态**（RAG 核查第 2 项）：**不能把故障显示成"教材里没有"**。
    #: ```
    #: {"state": "not_triggered"|"no_hit"|"failed"|"ok",
    #:  "reason": "…", "returned": 3, "rerank": "ok"|"fallback"|…,
    #:  "timings": {"embed_sec":…, "search_sec":…, "rerank_sec":…, "total_sec":…}}
    #: ```
    #: 页面据此如实说明"**这轮有没有用教材、卡在哪一步**"；
    #: ⚠️ 与审计状态（重放是否逐字一致）**是两套**，别混。
    rag_status: dict = {}
    node_timings_ms: dict[str, float] = {}

    #: ⚠️ **实际投递给模型的教材片段**（RAG 核查第 1 项：学生端可展开来源）。
    #: `[{source, page, block_index, type, text, truncated}, …]`
    #: ⚠️ 与 Tutor 收到的是**同一份**（同一个预算、同一套区间）；
    #: 不许让页面显示一段模型其实没拿到的"依据"。
    rag_sources: list[dict] = []
    rag_materials_in_prompt: bool = False
    subquestion_context: dict = {}


class ResetRequest(BaseModel):
    student_id: str
    problem_id: str


class FinishRequest(BaseModel):
    final_correct: bool | None = None


def validate_explicit_attempt(
    attempt_id: str, student_id: str, problem_id: str,
) -> AttemptRecord | None:
    """只读校验显式续做目标；空 ID 不创建作答，供媒体入口预检。

    调用方仍需在学生/题目锁内再次校验后才能更新状态，避免预检后终态变化。
    归属校验用于请求一致性，不替代学生身份认证。
    """
    if not attempt_id:
        return None
    record = get_attempt_store().load_attempt(attempt_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"attempt {attempt_id} 不存在")
    if record.student_id != student_id or record.problem_id != problem_id:
        raise HTTPException(status_code=403, detail="attempt 不属于请求指定的学生和题目")
    if record.status != STATUS_IN_PROGRESS:
        raise HTTPException(
            status_code=409,
            detail=f"attempt 已结束（{record.status}），请开启新的作答",
        )
    return record


def _resolve_attempt(req: AnalyzeRequest, problem: Problem | None) -> AttemptRecord:
    """定位本次作答属于哪个 Attempt（P11）。

    优先级：显式 attempt_id > 该学生该题未结束的 attempt > 新建一个。
    """
    store = get_attempt_store()
    if req.attempt_id:
        record = validate_explicit_attempt(req.attempt_id, req.student_id, req.problem_id)
        assert record is not None
        return record
    record = store.load_open_attempt(req.student_id, req.problem_id)
    if record is not None:
        return record
    return store.create_attempt(req.student_id, req.problem_id, problem=problem)


def _load_state(record: AttemptRecord, problem: Problem | None, override: str = "") -> StudentState:
    """取出 attempt 的 StudentState；补齐题目上下文（P11：状态由存储层提供）。

    题面优先级：请求显式传入 > 快照中已有 > 按 problem_id 从题库加载。
    """
    state = record.state or StudentState(
        student_id=record.student_id, problem_id=record.problem_id
    )
    if override:
        state.problem_content = override
    elif not state.problem_content and problem is not None:
        state.problem_content = problem.full_text
    return state


def _elapsed_seconds(record: AttemptRecord) -> float:
    """相对 attempt 开始的秒数（Learning Event 统一时间轴）。"""
    if record.started_at is None:
        return 0.0
    return round((time.time() - record.started_at.timestamp()), 1)


def _round_index(timeline: list[TimelineEvent]) -> int:
    """轮次编号 = 提交次数（只数 submit）。

    这里刻意不用 `len(timeline)`：timeline 还含 hint_request 等事件，
    用它当轮次会让"第几轮"随学生点了几次提示而跳号，trace 对不上轮次。
    """
    return sum(1 for event in timeline if event.event_type == "submit")


def _persist_failure_trace(
    *,
    attempt_id: str,
    round_index: int,
    student_input: str,
    request_id: str,
    problem_id: str,
    exc: Exception,
    usage,
    started: float,
    before_state: dict,
) -> None:
    """图执行抛异常时落一条失败 Trace（P20）。

    写库失败不影响主流程：这个函数的存在只是"尽量留下线索"，
    不能因为它自己出错而把原本要报的错盖掉。
    """
    try:
        persistence.persist_trace(
            attempt_id,
            round_index,
            payload={
                "student_input": student_input,
                "before_state": before_state,
                "graph_version": before_state.get("graph_version", ""),
                "graph_binding_status": before_state.get("graph_binding_status", "unknown"),
                "graph_release_status": before_state.get("graph_release_status", "unknown"),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            },
            request_id=request_id,
            problem_id=problem_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            llm_latency_ms=usage.latency_ms,
            provider="+".join(usage.providers),
            model=usage.model_label(),
            token_usage=usage.as_dict(),
            prompt_versions=prompt_version_fields(),
            before_state=before_state,
        )
    except Exception as nested:  # noqa: BLE001 落 trace 失败不能盖掉原始错误
        logger.warning("失败 Trace 也未能落库：%s: %s", type(nested).__name__, nested)


def _state_digest(state: StudentState) -> dict:
    """本轮开始前的状态摘要（P19 的 before_state）。

    只存排障真正需要对比的字段，不存整份 StudentState——trace 每轮一行，
    存两份完整快照会让表迅速膨胀。
    """
    return {
        "current_node": state.current_node,
        "graph_version": state.graph_version,
        "graph_binding_status": state.graph_binding_status,
        "graph_release_status": state.graph_release_status,
        "current_method": state.current_method,
        "visited_nodes": list(state.visited_nodes),
        "hint_level": state.hint_level,
        "consecutive_failures": state.consecutive_failures,
        "stuck_score": state.stuck_score,
        "weak_concepts": list(state.weak_concepts),
        "missing_reasoning": list(state.missing_reasoning),
        "node_attempts": dict(state.node_attempts),
        "node_errors": dict(state.node_errors),
        "last_hint_node": state.last_hint_node,
    }


# 同一学生同一题的并发请求最多等待多久（秒）。正常情况下请求是串行的，
# 只有双击提交 / 多标签页才会真的撞上；给足一个 LLM 轮次的时间即可。
ATTEMPT_LOCK_TIMEOUT = 90.0


@contextmanager
def _attempt_lock(store, student_id: str, problem_id: str):
    """序列化分析与生命周期变更，避免结束后的作答被迟到请求写回。"""
    lock = store.attempt_lock(student_id, problem_id)
    if not lock.acquire(timeout=ATTEMPT_LOCK_TIMEOUT):
        raise HTTPException(
            status_code=409,
            detail=(
                f"该学生该题（{student_id}/{problem_id}）上一次作答仍在处理中，"
                "请等待结果后再提交"
            ),
        )
    try:
        yield
    finally:
        lock.release()


def _delivered_rag_sources(retrieved_context) -> list[dict]:
    """**实际投递的教材片段**（薄封装）。

    ⚠️ 判定逻辑**只有一份**（在 `app.agent.graph.delivered_rag_sources`）——
    这里再写一遍就会出现"页面显示的来源 ≠ 模型看到的依据"（本项目已栽过多次）。
    """
    from app.agent.graph import delivered_rag_sources

    try:
        return delivered_rag_sources(retrieved_context)
    except Exception:  # noqa: BLE001 来源展示失败不该击穿批改结果
        return []


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_attempt(req: AnalyzeRequest) -> AnalyzeResponse:
    """分析学生当前作答：判断步骤、归因错误、检索知识、生成分级提示，并落库。

    整个「加载状态 → 跑工作流 → 写回」在**同一学生同一题**的互斥锁内完成，
    否则并发请求会各自基于旧快照写回，造成状态回退与提示丢失（见 AttemptStore.attempt_lock）。
    """
    store = get_attempt_store()
    with _attempt_lock(store, req.student_id, req.problem_id):
        return _analyze_locked(req, store)


def _analyze_locked(req: AnalyzeRequest, store, *, record_override=None, sandbox=False,
                    usage_capture=None) -> AnalyzeResponse:
    round_started = time.perf_counter()
    request_id = get_request_id()
    problem = load_problem(req.problem_id)
    graph: ProblemGraph | None = None

    # 题目不存在且调用方也没带题面 → 明确 404。
    # 否则会一路跑下去：学生在一个空题面上"作答"，还会因为外键约束失败
    # 只在内存里留下一个幽灵 attempt（查询接口看不到、日志却报"PostgreSQL 不可用"）。
    if problem is None and not req.problem_content.strip():
        raise HTTPException(
            status_code=404,
            detail=f"题目 {req.problem_id} 不在题库中（可改用 problem_content 直接传题面）",
        )

    record = record_override or _resolve_attempt(req, problem)
    state = _load_state(record, problem, override=req.problem_content)
    from app.services.problem_binding import question_snapshot
    from app.services.problem_repository import get_problem_repository
    if state.problem_binding_status == "unknown" and not state.timeline and problem is not None:
        state.problem_snapshot, state.problem_version = question_snapshot(problem, get_problem_repository())
        state.problem_binding_status = "bound"
    if state.problem_snapshot:
        from app.models.problem import Problem
        problem = Problem.model_validate(state.problem_snapshot)
        if req.problem_content.strip() and req.problem_content.strip() != problem.full_text:
            raise HTTPException(409, detail="当前作答已绑定原题，请新开作答修改题面")
        state.problem_content = problem.full_text

    hint_only = req.request_hint and not req.answer.strip()
    # A help request with no new writing refers to the persisted current answer,
    # including the latest OCR correction. Never replace it with UI placeholder text.
    if hint_only:
        req = req.model_copy(update={"answer": state.current_answer})
    previous_timeline_length = len(state.timeline)
    from app.services.graph_binding import bind_graph
    from app.services.problem_repository import get_problem_repository
    release = (get_problem_repository().graph_release(req.problem_id)
               if state.graph_binding_status in {"unbound", "graphless"}
               else {"graph": None, "graph_status": state.graph_release_status})
    try:
        graph = bind_graph(state, release["graph"], release["graph_status"])
        if graph is not None and graph.metadata.get("question_version") and state.problem_version:
            if graph.metadata["question_version"] != state.problem_version:
                raise ValueError("Graph与本次作答绑定的题目版本不一致，请核对发布版本")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    graph_events = state.timeline[previous_timeline_length:]
    elapsed = _elapsed_seconds(record)
    for event in graph_events:
        event.timestamp = elapsed
    # Persist the first selected version before an expensive or failing model run.
    record = store.save_attempt(record, state)
    if graph_events:
        state.timeline = state.timeline[:previous_timeline_length]
    before_state = _state_digest(state)
    before_record = record.model_dump(mode="json")
    before_record["state"] = state.model_dump(mode="json", exclude={"media_sources"})

    # ---- 计数：上轮累计值来自 attempt（跨请求持久）----
    error_count = record.error_count
    hint_count = record.hint_count
    stuck_count = record.stuck_count
    if req.request_hint:
        hint_count += 1

    # ---- P4：图感知卡顿检测（行为规则分 + Graph 信号加成）----
    signals = StuckSignals(
        idle_time=req.idle_time,
        error_count=error_count,
        rewrite_count=req.rewrite_count,
        hint_count=hint_count,
    )
    graph_signals = build_graph_signals(graph, state)
    stuck_score = compute_graph_aware_stuck_score(signals, graph_signals)
    reasons = stuck_reasons(signals, graph_signals)

    new_events = [*graph_events]
    if not hint_only:
        new_events.append(TimelineEvent(timestamp=elapsed, event_type="submit", content=req.answer))
    if req.request_hint:
        new_events.append(
            TimelineEvent(
                timestamp=elapsed,
                event_type="hint_request",
                content="学生主动请求提示",  # 具体级别由图内 Hint Policy 在本轮决定
            )
        )
    timeline = [*state.timeline, *new_events]

    # ---- P6：状态驱动工作流（Step Matching / 诊断 / RAG / Hint Policy 均在图内条件路由）----
    graph_input = {
        "student_id": req.student_id,
        "problem_id": req.problem_id,
        "problem_content": state.problem_content,
        "current_answer": req.answer,
        "request_hint": req.request_hint,
        "graph_mode": graph is not None,
        "graph_binding_status": state.graph_binding_status,
        "graph_version": state.graph_version,
        "graph_snapshot": state.graph_snapshot,
        "graph_release_status": state.graph_release_status,
        "hint_level": state.hint_level,      # 上一轮下发的级别（策略输入）
        "stuck_score": stuck_score,
        "timeline": timeline,
        "profile_policy_version": "paper_evidence_v1",
        "legacy_weak_concepts": sorted(set(state.legacy_weak_concepts + (
            state.weak_concepts if state.profile_policy_version != "paper_evidence_v1" else []))),
        "weak_concepts": (state.weak_concepts if state.profile_policy_version == "paper_evidence_v1" else []),
        "error_evidence": state.error_evidence,
        "help_events": state.help_events,
        "hint_history": state.hint_history,
        "visited_nodes": state.visited_nodes,
        "current_node": state.current_node,
        # 上一轮的人类可读步骤名。必须传进去：答对轮会跳过 analyze_step，
        # 若这里不传，`current_step` 会保持模型默认值 ""，
        # 响应就成了"current_node=S3 但 current_step 为空"，展示层只剩一个编号。
        "current_step": state.current_step,
        "current_method": state.current_method,
        "node_attempts": state.node_attempts,
        "node_errors": state.node_errors,
        "missing_reasoning": state.missing_reasoning,
        "consecutive_failures": state.consecutive_failures,
        "last_hint_node": state.last_hint_node,
    }
    # Problem / ProblemGraph 通过 configurable 注入图节点
    # P19：在归集器内跑工作流，本轮所有 LLM 调用的 model / token / 延迟都会被收集
    with track_usage() as usage:
        if usage_capture is not None:
            usage_capture.append(usage)
        try:
            result = tutor_graph.invoke(
                graph_input,
                config={
                    "configurable": {
                        "problem": problem,
                        "problem_graph": graph,
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001 P20：图内异常也要留下可排障的 Trace
            # 正常情况下节点级降级已经拦住了 LLM 故障，能走到这里说明是**代码缺陷**。
            # 此时最要紧的是别让这一轮凭空消失——恰恰是最需要排障的那一轮，
            # 若连 Trace 都没有，就只能靠猜。落一条失败记录后再如实报错。
            if not sandbox:
                _persist_failure_trace(
                    # 注意用 record.attempt_id：函数内那个 attempt_id 是在图跑完之后才赋值的，
                    # 异常路径上它还不存在。
                    attempt_id=record.attempt_id,
                    round_index=_round_index(timeline),
                    student_input=req.answer,
                    request_id=request_id,
                    problem_id=req.problem_id,
                    exc=exc,
                    usage=usage,
                    started=round_started,
                    before_state=before_state,
                )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"本轮分析失败（{type(exc).__name__}）：{exc}。"
                    "该轮已记入 trace，可用 GET /attempt/{attempt_id}/traces 排查。"
                ),
            ) from exc

    if result.get("has_obvious_error"):
        error_count += 1
    if is_stuck(stuck_score):
        stuck_count += 1

    new_state = StudentState(**result)
    for field in ("graph_version", "graph_snapshot", "graph_binding_status", "graph_release_status",
                  "problem_version", "problem_snapshot", "problem_binding_status"):
        setattr(new_state, field, getattr(state, field))
    new_state.timeline = timeline
    if not sandbox:
        from app.services.ocr_correction import finalize_media_profile
        finalize_media_profile(record)
    new_state.media_sources = list(record.state.media_sources if record.state else state.media_sources)
    source_id = uuid.uuid4().hex if req._media_type else ""
    for evidence in new_state.error_evidence[len(state.error_evidence):]:
        evidence.update(source_id=source_id, source_revision=0,
                        request_id=request_id, problem_id=req.problem_id)
    if source_id:
        new_state.media_sources.append({
            "source_id": source_id, "media_type": req._media_type,
            "timestamp": req._media_timestamp, "original_text": req.answer,
            "revision": 0, "before_record": before_record,
            "request": req.model_dump(), "timeline_length": len(timeline),
            "revisions": [], "profile_finalized": False,
        })

    # ---- P11：状态写回 attempt（投影 + 完整快照 + 计数）----
    record = store.save_attempt(
        record,
        new_state,
        error_count=error_count,
        hint_count=hint_count,
        stuck_count=stuck_count,
    )
    attempt_id = record.attempt_id
    status = record.status

    if not sandbox:
        # ---- P22：回看判定上一条提示有没有用 ----
        #
        # 放在这里（本轮图已跑完、新提示还没落库）：此时**上一轮提示的效果**
        # 已经能从状态差分看出来，而本轮的新提示还不存在，不会自己判自己。
        resolved = 0 if req._media_type else persistence.resolve_pending_hints(
            attempt_id,
            visited_after=list(new_state.visited_nodes or []),
            current_node_after=new_state.current_node,
            correct_after=bool(new_state.round_correct),
            consecutive_failures_after=int(new_state.consecutive_failures or 0),
            off_graph_after=bool(new_state.round_off_graph),
            degraded_after=bool(new_state.degraded_reasons),
        )
        if resolved:
            logger.info("P22：判定 %d 条提示效果", resolved)

        # ---- Phase 6：交互落库（DB 不可用时内部自动降级）----
        persistence.persist_events(attempt_id, new_events)
        if result.get("hint"):
            persistence.persist_hint(
                attempt_id,
                hint_level=new_state.hint_level,
                hint=result["hint"],
                issue_type=result.get("issue_type", ""),
                # P22：判定"有没有推进"必须知道**基准在哪**——
                # 没有提示前的状态快照，之后的变化就无从比较。
                node=new_state.current_node,
                state_before={
                    "visited_nodes": list(new_state.visited_nodes or []),
                    "current_node": new_state.current_node,
                    "consecutive_failures": int(new_state.consecutive_failures or 0),
                    "hint_level": int(new_state.hint_level or 0),
                    "source_id": source_id,
                    "source_revision": 0,
                },
            )
            # ---- P11：提示历史进 Attempt 投影（跨请求累积，重启后可回看）----
            store.update_attempt(
                attempt_id,
                hint_history=[
                    *record.hint_history,
                    {
                        "hint_level": new_state.hint_level,
                        "hint": result["hint"],
                        "issue_type": result.get("issue_type", ""),
                        "node": new_state.current_node,
                        "source_id": source_id,
                        "source_revision": 0,
                    },
                ],
            )
        if not req._media_type:
            persistence.persist_profile_weak_concepts(
                req.student_id, result.get("weak_concepts", [])
            )
        # ---- P9/P19：每轮完整 Tracing 落库（DB 不可用时静默降级）----
        prompt_versions = prompt_version_fields()
        persistence.persist_trace(
            attempt_id,
            round_index=_round_index(timeline),
            request_id=request_id,
            problem_id=req.problem_id,
            latency_ms=int((time.perf_counter() - round_started) * 1000),
            llm_latency_ms=usage.latency_ms,
            provider="+".join(usage.providers),
            model=usage.model_label(),
            token_usage=usage.as_dict(),
            prompt_versions=prompt_versions,
            before_state=before_state,
            payload={
                "student_input": req.answer,
                "input_mode": "hint_only" if hint_only else "new_answer",
                "node_timings_ms": dict(usage.node_timings_ms),
                "subquestion_context": dict(result.get('subquestion_context') or {}),
                "source_id": source_id,
                "source_revision": 0,
                # 本轮产生的 Learning Event（P19：输入来源可替换，事件是统一协议）
                "learning_events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "content": event.content,
                    }
                    for event in new_events
                ],
                # P23：记下**本轮实际**运行模式。不能靠 before_state.graph_mode——
                # 那是状态里上一轮的残留值；这道题中途补上图后，同一 attempt 会从
                # graphless 自动切到 graph 模式，切换点要能在这里直接看出来。
                "graph_mode": graph is not None,
                "graph_version": state.graph_version,
                "graph_binding_status": state.graph_binding_status,
                "graph_release_status": state.graph_release_status,
                # P20：本轮降级原因（空 = 一切正常）。排障时先看这个字段，
                # 它能直接区分"模型答错了"与"模型根本没答上来"。
                "degraded_reasons": list(result.get("degraded_reasons") or []),
                "step_match": result.get("last_step_match") or {},
                "rag_query": result.get("rag_query", ""),
                "rag_context": result.get("retrieved_context", []),
                # ⚠️ **本轮到底有没有用上教材、卡在哪一步**（RAG 核查第 2 项）。
                #    没有这个字段，Trace 里"空 context"分不出
                #    「检索失败」与「教材里确实没有」—— 那是**两种完全不同的事**。
                #    ⚠️ 与审计状态（重放是否逐字一致）**是两套**，别混。
                "rag_status": result.get("rag_status") or {},
                "stuck_score": stuck_score,
                "stuck_reasons": reasons,
                "diagnosis": {
                    "issue_type": result.get("issue_type", ""),
                    "diagnosed_issue": result.get("diagnosed_issue", ""),
                    "severity": result.get("severity", ""),
                    "recommended_action": result.get("recommended_action", ""),
                },
                # P19：提示策略的完整决策（谁决定了这个级别、依据是什么）
                # 注意：这几个"本轮"字段要取 new_state（本轮结束后的状态）。
                # 取 state（本轮开始前）会让 trace 的 correct/off_graph 整整错开一轮——
                # 表现为 reasons 写着"本轮推进正确"而 correct 却是 False。
                "hint_policy": {
                    "level": result.get("hint_level", 0),
                    "reasons": result.get("hint_reasons", []),
                    "correct": new_state.round_correct,
                    "off_graph": new_state.round_off_graph,
                    "consecutive_failures": new_state.consecutive_failures,
                    # 取策略节点回传的**入参原值**，不要用 new_state 重算：
                    # update_state 之后 last_hint_node 已被改写成本轮节点，
                    # 重算几乎恒为 True，与 hint_reasons 自相矛盾。
                    "same_hint_before": bool(result.get("hint_same_before", False)),
                },
                "hint_level": result.get("hint_level", 0),
                "hint_reasons": result.get("hint_reasons", []),
                "tutor_output": result.get("hint", ""),
                "after_state": _state_digest(new_state),
                # ---- P13：记录本轮实际使用的提示词版本，便于 A/B 对比与回归定位 ----
                **prompt_versions,
            },
        )

    # ---- P11：答完整题即结束 attempt 生命周期 ----
    if result.get("finished"):
        store.finish_attempt(attempt_id, final_correct=True, status=STATUS_FINISHED)
        status = STATUS_FINISHED

    graph_mode = result.get("graph_mode", False)
    response = AnalyzeResponse(
        attempt_id=attempt_id,
        source_id=source_id,
        node_timings_ms=dict(usage.node_timings_ms),
        current_step=result.get("current_step", ""),
        current_node=result.get("current_node", ""),
        visited_nodes=result.get("visited_nodes", []),
        graph_mode=graph_mode,
        problem_version=new_state.problem_version,
        problem_binding_status=new_state.problem_binding_status,
        graph_version=new_state.graph_version,
        graph_binding_status=new_state.graph_binding_status,
        graph_release_status=new_state.graph_release_status,
        status=status,
        issue=result.get("diagnosed_issue", ""),
        issue_type=result.get("issue_type", ""),
        severity=result.get("severity", ""),
        stuck_score=result.get("stuck_score", stuck_score),
        weak_concepts=result.get("weak_concepts", []),
        hint_level=result.get("hint_level", new_state.hint_level),
        hint=result.get("hint", ""),
        finished=result.get("finished", False),
        step_match=result.get("last_step_match") or None,
        stuck_reasons=reasons,
        hint_reasons=result.get("hint_reasons", []),
        # P23：无 Graph 时额外给一份阶段级判断，让 graphless 不是"残缺版 graph 模式"
        graphless=None if graph_mode else build_assessment(result),
        # P20：降级信息如实上报，前端据此提示"本轮分析暂时不可用"
        degraded=bool(result.get("degraded_reasons")),
        degraded_reasons=list(result.get("degraded_reasons") or []),
        # 安全替换**单独上报**：它不改变 degraded，也不改变可评性（用户裁决）。
        safety_substitutions=list(result.get("safety_substitutions") or []),
        rag_status=dict(result.get("rag_status") or {}),
        rag_sources=(_delivered_rag_sources(result.get("retrieved_context"))
                     if result.get('rag_materials_in_prompt') else []),
        rag_materials_in_prompt=bool(result.get('rag_materials_in_prompt')),
        hint_optional=bool(result.get('hint_optional')),
        hint_evidence=list(result.get('hint_evidence') or []),
        diagnosis_text_kind=result.get('diagnosis_text_kind','legacy'),
        subquestion_context=dict(result.get('subquestion_context') or {}),
    )
    if source_id:
        latest = store.load_attempt(attempt_id)
        latest.state.media_sources[-1]["revisions"] = [
            {"revision": 0, "text": req.answer, "analysis": response.model_dump()}
        ]
        store.save_attempt(latest, latest.state)
        if response.finished and not sandbox:
            from app.services.ocr_correction import finalize_media_profile
            finalize_media_profile(latest)
            store.save_attempt(latest, latest.state)
    return response


@router.post("/reset")
def reset_attempt(req: ResetRequest) -> dict:
    """重置某学生某题的作答会话：结束当前 attempt（保留历史），下次作答开启新的。"""
    store = get_attempt_store()
    with _attempt_lock(store, req.student_id, req.problem_id):
        record = store.load_open_attempt(req.student_id, req.problem_id)
        if record is not None:
            from app.services.ocr_correction import finalize_media_profile
            finalize_media_profile(record)
            store.save_attempt(record, record.state)
            store.abandon_attempt(record.attempt_id)
    return {
        "status": "reset",
        "student_id": req.student_id,
        "problem_id": req.problem_id,
        "abandoned_attempt_id": record.attempt_id if record else None,
    }


@router.post("/{attempt_id}/finish")
def finish_attempt_endpoint(attempt_id: str, req: FinishRequest | None = None) -> dict:
    """显式结束一次作答（P11 生命周期出口）。"""
    store = get_attempt_store()
    record = store.load_attempt(attempt_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"attempt {attempt_id} 不存在")
    final_correct = req.final_correct if req else None
    with _attempt_lock(store, record.student_id, record.problem_id):
        # 等待分析释放锁后重读；终态只返回既有结果，不覆盖完成判定或复活放弃记录。
        record = store.load_attempt(attempt_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"attempt {attempt_id} 不存在")
        if record.status == STATUS_FINISHED:
            return {"attempt_id": attempt_id, "status": record.status, "final_correct": record.final_correct}
        if record.status != STATUS_IN_PROGRESS:
            raise HTTPException(status_code=409, detail=f"attempt 已结束（{record.status}）")
        from app.services.ocr_correction import finalize_media_profile
        finalize_media_profile(record)
        store.save_attempt(record, record.state)
        store.finish_attempt(attempt_id, final_correct=final_correct, status=STATUS_FINISHED)
    return {"attempt_id": attempt_id, "status": STATUS_FINISHED, "final_correct": final_correct}


@router.get("/{attempt_id}")
def get_attempt(attempt_id: str) -> dict:
    """查询完整作答记录：attempt 概要 + 学生状态 + Learning Timeline + 提示历史。"""
    from app.services.problem_binding import student_view
    unavailable = None
    try:
        detail = persistence.get_attempt_detail(attempt_id)
    except persistence.PersistenceUnavailable as exc:
        unavailable = exc
        detail = None
    if detail is not None:
        return student_view(detail)

    # 降级期间的作答已保存在 AttemptStore，切题恢复不能只查数据库。
    # 快照中的时间线与提示历史是真实记录，但不冒充数据库事件表。
    record = get_attempt_store().load_attempt(attempt_id)
    if record is not None:
        result = record.model_dump(mode="json", exclude={
            "started_at", "updated_at", "finished_at",
        })
        result.update(
            start_time=record.started_at.isoformat() if record.started_at else None,
            end_time=record.finished_at.isoformat() if record.finished_at else None,
            events=[event.model_dump(mode="json")
                    for event in (record.state.timeline if record.state else [])],
            hints=list(record.hint_history),
            record_source="state_snapshot",
        )
        return student_view(result)
    if unavailable is not None:
        raise HTTPException(status_code=503, detail=f"PostgreSQL 不可用：{unavailable}") from unavailable
    raise HTTPException(status_code=404, detail=f"attempt {attempt_id} 不存在")


@router.get("/{attempt_id}/traces")
def get_attempt_traces(attempt_id: str) -> dict:
    """逐轮完整 Tracing（P19）：定位是哪一环出错，而不是把系统当黑盒。

    每一轮包含：request_id / problem_id / 学生输入 / Learning Event / before_state /
    StepMatch / 诊断 / RAG 查询与上下文 / Hint Policy 决策 / Tutor 输出 /
    after_state / 延迟 / token 用量 / 模型 / 提示词版本。
    """
    try:
        if persistence.get_attempt_row(attempt_id) is None:
            raise HTTPException(status_code=404, detail=f"attempt {attempt_id} 不存在")
        traces = persistence.get_traces(attempt_id)
    except persistence.PersistenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"PostgreSQL 不可用：{exc}") from exc
    return {"attempt_id": attempt_id, "count": len(traces), "traces": traces}
