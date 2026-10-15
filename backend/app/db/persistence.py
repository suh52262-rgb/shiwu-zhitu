"""PostgreSQL 持久化层（Phase 6）。

原则：**数据库不可用时全部静默降级**——写入类函数返回 False 并记录一次警告，
查询类函数抛出 PersistenceUnavailable 由 API 层转 503。
教学闭环不因基础设施缺席而中断（内存中仍有会话热缓存）。
"""
import functools
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.postgres import get_engine
from app.db.tables import (
    AnalysisTrace,
    Attempt,
    HintRecord,
    LearningEvent,
    Problem,
    Student,
    StudentProfile,
)
from app.models.attempt import STATUS_FINISHED, STATUS_IN_PROGRESS
from app.models.problem import Problem as ProblemSchema
from app.models.student import TimelineEvent

logger = logging.getLogger(__name__)

# 可由 save_attempt_state 更新的 attempts 列（白名单，避免任意字段写入）
ATTEMPT_MUTABLE_FIELDS = frozenset(
    {
        "status",
        "current_node",
        "current_method",
        "visited_nodes",
        "mistakes",
        "missing_reasoning",
        "weak_concepts",
        "stuck_score",
        "hint_history",
        "error_count",
        "hint_count",
        "stuck_count",
        "final_correct",
        "state",
    }
)


class PersistenceUnavailable(RuntimeError):
    """PostgreSQL 不可达。"""


# 同类错误只警告一次，避免每次请求刷屏
_warned_keys: set[str] = set()


# ---------------------------------------------------------------- 熔断器
#
# 为什么需要：数据库不可用时，**每一次**读写都要现付一次连接超时。
# 实测：postgres_connect_timeout=3s，而 localhost 同时解析到 ::1 与 127.0.0.1，
# psycopg 两个都试，于是单次耗时 6 秒。
#
# 后果不是"降级"，而是"变成每个请求白等 6 秒的慢服务"——
# 一轮 /attempt/analyze 会调用多次写库，一轮就是十几秒。
#
# 熔断器把"数据库挂了"这个结论记住一小段时间，期间直接走降级、
# 不再尝试连接；冷却结束再放一次请求去探活（探活成功即恢复）。
# 这样数据库真的恢复时后端会自动恢复，不需要重启。
_BREAKER_COOLDOWN_SECONDS = 10.0
_breaker_open_until: float = 0.0


def _breaker_allows() -> bool:
    """熔断器是否放行（放行 = 可以尝试连接数据库）。"""
    global _breaker_open_until
    if time.monotonic() >= _breaker_open_until:
        # 冷却结束：清零并放行。成功则一直放行；失败会经 _warn_once 重新熔断。
        _breaker_open_until = 0.0
        return True
    return False


def _breaker_trip() -> None:
    """标记数据库不可用，冷却期内不再尝试。"""
    global _breaker_open_until
    _breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS


def breaker_status() -> dict:
    """熔断器状态（供 /health 或排障查看）。"""
    remaining = max(0.0, _breaker_open_until - time.monotonic())
    return {"open": remaining > 0, "resets_in_seconds": round(remaining, 1)}


def _db_guarded(*, raises: bool, no_db_when=None):
    """熔断器入口闸门。

    打开时**直接返回降级结果**，不再去碰数据库——这正是省下那 6 秒的地方。
    冷却期一过自动放行一次；放行后的失败会经 `_warn_once` 再次熔断，
    成功则自然保持放行，因此数据库恢复后不需要重启。

    `raises=True`：读操作，降级时抛 `PersistenceUnavailable`（调用方要能区分"没有"与"查不到"）
    `raises=False`：写操作，静默降级返回 False（教学闭环不该因为落库失败而中断）
    `no_db_when`：谓词，为真表示**这次调用根本不需要数据库**（例如空事件列表是 no-op）。
        这类调用不该被熔断影响——把它算成降级是错的。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if no_db_when is not None and no_db_when(*args, **kwargs):
                return func(*args, **kwargs)      # no-op 路径：与数据库状态无关
            if not _breaker_allows():
                if raises:
                    raise PersistenceUnavailable(
                        f"PostgreSQL 不可用（熔断中，{breaker_status()['resets_in_seconds']}s 后重试）"
                    )
                return False
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _warn_once(key: str, exc: Exception) -> None:
    # 指纹里带上异常类型：原来所有 DB 错误共用常量键 "db"，进程内第一次告警
    # （可能只是瞬时超时）之后，**性质完全不同的后续错误都不再出声**——
    # 包括真正需要修的 "column does not exist"。
    _breaker_trip()          # 所有失败路径都会经过这里，正好作为熔断触发点
    fingerprint = f"{key}:{type(exc).__name__}"
    if fingerprint not in _warned_keys:
        _warned_keys.add(fingerprint)
        logger.warning("PostgreSQL 不可用，持久化降级为内存模式：%s", exc)
    else:
        logger.debug("PostgreSQL 仍不可用（%s）", fingerprint)


# ---------- 写入 ----------

@_db_guarded(raises=False)
def persist_attempt_start(
    student_id: str,
    problem_id: str,
    attempt_id: str,
    problem: ProblemSchema | None = None,
) -> bool:
    """幂等写入学生 / 题目 / attempt 三行（attempt 已存在则原样保留）。"""
    try:
        with Session(get_engine()) as session, session.begin():
            session.merge(Student(id=student_id))
            if problem is not None:
                session.merge(
                    Problem(
                        id=problem.problem_id,
                        title=problem.title,
                        content=problem.content,
                        knowledge_points=problem.knowledge_points,
                        difficulty=problem.difficulty,
                        answer=problem.answer,
                        extra=problem.metadata,
                    )
                )
            if session.get(Attempt, attempt_id) is None:
                session.add(
                    Attempt(
                        id=attempt_id,
                        student_id=student_id,
                        problem_id=problem_id,
                        status=STATUS_IN_PROGRESS,
                        error_count=0,
                        hint_count=0,
                        stuck_count=0,
                        current_node="",
                        current_method="",
                        visited_nodes=[],
                        mistakes=[],
                        missing_reasoning=[],
                        weak_concepts=[],
                        hint_history=[],
                        stuck_score=0.0,
                    )
                )
        return True
    except Exception as exc:  # noqa: BLE001 任何 DB 异常都降级
        _warn_once("db", exc)
        return False


@_db_guarded(raises=False, no_db_when=lambda attempt_id, events: not events)
def persist_events(attempt_id: str, events: list[TimelineEvent]) -> bool:
    """批量写入 Learning Events。"""
    if not events:
        return True
    try:
        with Session(get_engine()) as session, session.begin():
            session.add_all(
                LearningEvent(
                    attempt_id=attempt_id,
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    content=event.content,
                )
                for event in events
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


@_db_guarded(raises=False)
def persist_hint(attempt_id: str, hint_level: int, hint: str, issue_type: str,
                 *, node: str = "", state_before: dict | None = None) -> int | None:
    """记录一次下发的提示，**返回 hint_id**（失败返回 None）。

    P22 起返回 id 而不是 bool：效果判定要**回写这一条**，
    拿不到 id 就只能靠"最新一条"猜，并发或补写时会写错行。

    调用方若只做真假判断，`int` 与 `bool` 的用法兼容（id 从 1 起，
    失败是 None，两者真假性一致）。
    """
    try:
        with Session(get_engine()) as session, session.begin():
            row = HintRecord(
                attempt_id=attempt_id,
                hint_level=hint_level,
                hint=hint,
                issue_type=issue_type,
                node=node,
                state_before=state_before or {},
            )
            session.add(row)
            session.flush()          # 取到自增 id
            return int(row.id)
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return None


@_db_guarded(raises=False)
def resolve_pending_hints(attempt_id: str, *, visited_after: list[str],
                          current_node_after: str, correct_after: bool,
                          consecutive_failures_after: int,
                          off_graph_after: bool = False,
                          degraded_after: bool = False,
                          exclude_source_id: str = "") -> int:
    """回看并判定该 attempt 上**所有还没判定的提示**。返回判定条数。

    在**每轮开始时**调用：此时上一轮提示的效果已经能从状态差分看出来了。

    判定逻辑在 `app/agent/hint_effect.py`（纯函数，可测），
    这里只负责取数据、写结论——**不让 I/O 与判断缠在一起**。
    """
    from datetime import datetime, timezone

    from app.agent.hint_effect import judge_hint_effect

    if not attempt_id:
        return 0
    try:
        with Session(get_engine()) as session, session.begin():
            rows = session.execute(
                select(HintRecord)
                .where(HintRecord.attempt_id == attempt_id,
                       HintRecord.resolved_at.is_(None))
                .order_by(HintRecord.id)
            ).scalars().all()
            resolved_count = 0
            for row in rows:
                snapshot = row.state_before or {}
                if exclude_source_id and snapshot.get("source_id") == exclude_source_id:
                    continue
                effect = judge_hint_effect(
                    hint_node=row.node,
                    visited_before=list(snapshot.get("visited_nodes") or []),
                    visited_after=list(visited_after or []),
                    current_node_after=current_node_after,
                    correct_after=correct_after,
                    consecutive_failures_after=consecutive_failures_after,
                    off_graph_after=off_graph_after,
                    degraded_after=degraded_after,
                )
                row.effective = effect.effective
                row.effective_evidence = effect.evidence
                row.resolved_at = datetime.now(timezone.utc)
                resolved_count += 1
            return resolved_count
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return 0


@_db_guarded(raises=False)
def persist_attempt_finish(
    attempt_id: str,
    final_correct: bool | None,
    hint_count: int,
    stuck_count: int,
) -> bool:
    """更新 attempt 计数；完成后写入 final_correct 与 end_time。"""
    try:
        with Session(get_engine()) as session, session.begin():
            row = session.get(Attempt, attempt_id)
            if row is None:
                return False
            row.hint_count = hint_count
            row.stuck_count = stuck_count
            if final_correct is True and row.final_correct is not True:
                row.final_correct = True
                row.end_time = datetime.now(timezone.utc)
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


# ---------- Attempt 生命周期（P11） ----------

def _attempt_row_to_dict(row: Attempt) -> dict:
    """ORM 行 → 普通 dict（None 归一为默认值），供 services/attempt_store 映射。"""
    return {
        "attempt_id": row.id,
        "student_id": row.student_id,
        "problem_id": row.problem_id,
        "status": row.status or STATUS_IN_PROGRESS,
        "current_node": row.current_node or "",
        "current_method": row.current_method or "",
        "visited_nodes": list(row.visited_nodes or []),
        "mistakes": list(row.mistakes or []),
        "missing_reasoning": list(row.missing_reasoning or []),
        "weak_concepts": list(row.weak_concepts or []),
        "stuck_score": float(row.stuck_score or 0.0),
        "hint_history": list(row.hint_history or []),
        "error_count": int(row.error_count or 0),
        "hint_count": int(row.hint_count or 0),
        "stuck_count": int(row.stuck_count or 0),
        "started_at": row.start_time,
        "updated_at": row.updated_at,
        "finished_at": row.end_time,
        "final_correct": row.final_correct,
        "state": row.state,
    }


@_db_guarded(raises=False)
def save_attempt_state(attempt_id: str, **fields: object) -> bool:
    """部分更新 attempt 的状态投影 / 计数 / 完整快照。

    字段名须在 `ATTEMPT_MUTABLE_FIELDS` 白名单内，其余忽略（防止误写）。
    返回 False 表示数据库不可用或 attempt 不存在——调用方据此降级。
    """
    updates = {key: value for key, value in fields.items() if key in ATTEMPT_MUTABLE_FIELDS}
    if not updates:
        return True
    try:
        with Session(get_engine()) as session, session.begin():
            row = session.get(Attempt, attempt_id)
            if row is None:
                return False
            for key, value in updates.items():
                setattr(row, key, value)
            row.updated_at = datetime.now(timezone.utc)
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


@_db_guarded(raises=False)
def finish_attempt(
    attempt_id: str,
    final_correct: bool | None = None,
    status: str = STATUS_FINISHED,
) -> bool:
    """结束一次作答（完成或放弃），写入终态与结束时间。

    与 `persist_attempt_finish` 的区别：后者只累计计数、且仅在答对时才收尾；
    本函数用于显式结束生命周期（P11），放弃也会落 end_time。
    """
    try:
        with Session(get_engine()) as session, session.begin():
            row = session.get(Attempt, attempt_id)
            if row is None:
                return False
            row.status = status
            if final_correct is not None:
                row.final_correct = final_correct
            if row.end_time is None:
                row.end_time = datetime.now(timezone.utc)
            row.updated_at = datetime.now(timezone.utc)
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


@_db_guarded(raises=True)
def get_attempt_row(attempt_id: str) -> dict | None:
    """读取单个 attempt（含状态快照）。不存在返回 None；DB 不可达抛 PersistenceUnavailable。"""
    try:
        with Session(get_engine()) as session:
            row = session.get(Attempt, attempt_id)
            return _attempt_row_to_dict(row) if row is not None else None
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        raise PersistenceUnavailable(str(exc)) from exc


@_db_guarded(raises=True)
def get_open_attempt_row(student_id: str, problem_id: str) -> dict | None:
    """取该学生该题**最近一次未结束**的 attempt（用于服务重启后续做）。"""
    try:
        with Session(get_engine()) as session:
            row = session.scalars(
                select(Attempt)
                .where(
                    Attempt.student_id == student_id,
                    Attempt.problem_id == problem_id,
                    Attempt.status == STATUS_IN_PROGRESS,
                )
                .order_by(Attempt.start_time.desc(), Attempt.id.desc())
                .limit(1)
            ).first()
            return _attempt_row_to_dict(row) if row is not None else None
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        raise PersistenceUnavailable(str(exc)) from exc


@_db_guarded(raises=False)
def persist_profile_weak_concepts(student_id: str, weak_concepts: list[str]) -> bool:
    """把本轮诊断出的薄弱概念合并进学生画像（并集，不重复）。"""
    if not weak_concepts:
        return True
    try:
        with Session(get_engine()) as session, session.begin():
            row = session.get(StudentProfile, student_id)
            if row is None:
                row = StudentProfile(student_id=student_id, weak_concepts=[])
                session.add(row)
            merged = list(row.weak_concepts or [])
            for concept in weak_concepts:
                if concept not in merged:
                    merged.append(concept)
            row.weak_concepts = merged
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


@_db_guarded(raises=False)
def persist_trace(
    attempt_id: str,
    round_index: int,
    payload: dict,
    *,
    request_id: str = "",
    problem_id: str = "",
    latency_ms: int = 0,
    llm_latency_ms: int = 0,
    provider: str = "",
    model: str = "",
    token_usage: dict | None = None,
    prompt_versions: dict | None = None,
    before_state: dict | None = None,
) -> bool:
    """写入一轮分析的完整 Tracing（P9 建立，P19 补齐可观测性字段）。

    除 payload 外的参数都有默认值，既有调用方无需改动。
    DB 不可用时静默降级返回 False。
    """
    try:
        with Session(get_engine()) as session, session.begin():
            session.add(
                AnalysisTrace(
                    attempt_id=attempt_id,
                    round_index=round_index,
                    request_id=request_id,
                    problem_id=problem_id,
                    latency_ms=latency_ms,
                    llm_latency_ms=llm_latency_ms,
                    provider=provider,
                    model=model,
                    token_usage=token_usage or {},
                    prompt_versions=prompt_versions or {},
                    before_state=before_state or {},
                    payload=payload,
                )
            )
        return True
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        return False


@_db_guarded(raises=True)
def get_traces(attempt_id: str) -> list[dict]:
    """按轮次返回某 attempt 的全部 Tracing。DB 不可达抛 PersistenceUnavailable。"""
    try:
        with Session(get_engine()) as session:
            rows = session.scalars(
                select(AnalysisTrace)
                .where(AnalysisTrace.attempt_id == attempt_id)
                .order_by(AnalysisTrace.id)
            ).all()
            return [
                {
                    "round_index": row.round_index,
                    "request_id": row.request_id or "",
                    "problem_id": row.problem_id or "",
                    "latency_ms": row.latency_ms or 0,
                    "llm_latency_ms": row.llm_latency_ms or 0,
                    "provider": row.provider or "",
                    "model": row.model or "",
                    "token_usage": row.token_usage or {},
                    "prompt_versions": row.prompt_versions or {},
                    "before_state": row.before_state or {},
                    "payload": row.payload,
                    # created_at 有 server_default，但迁移新增的列在旧行上可能为 NULL，
                    # 无守卫会让 AttributeError 被包装成误导性的 503
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        raise PersistenceUnavailable(str(exc)) from exc


# ---------- 查询 ----------

@_db_guarded(raises=True)
def get_student_profile(student_id: str) -> dict | None:
    """学生画像 + 全部作答历史。学生不存在返回 None；DB 不可达抛 PersistenceUnavailable。"""
    try:
        with Session(get_engine()) as session:
            if session.get(Student, student_id) is None:
                return None
            profile = session.get(StudentProfile, student_id)
            attempts = session.scalars(
                select(Attempt)
                .where(Attempt.student_id == student_id)
                .order_by(Attempt.start_time)
            ).all()
            from app.services.learning_profile import current_profile_signals
            signals = current_profile_signals([(a.id, a.state) for a in attempts],
                                      (profile.weak_concepts if profile else []) or [],student_id)
            return {
                **signals,
                "student_id": student_id,
                "weak_concepts": (profile.weak_concepts if profile else []) or [],
                "attempts": [
                    {
                        "attempt_id": attempt.id,
                        "problem_id": attempt.problem_id,
                        "start_time": attempt.start_time.isoformat() if attempt.start_time else None,
                        "end_time": attempt.end_time.isoformat() if attempt.end_time else None,
                        "final_correct": attempt.final_correct,
                        "hint_count": attempt.hint_count,
                        "stuck_count": attempt.stuck_count,
                    }
                    for attempt in attempts
                ],
            }
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        raise PersistenceUnavailable(str(exc)) from exc


@_db_guarded(raises=True)
def get_attempt_detail(attempt_id: str) -> dict | None:
    """读取完整作答记录（attempt + learning_events + hint_records）。

    返回 None 表示记录不存在；PostgreSQL 不可达时抛 PersistenceUnavailable。
    """
    try:
        with Session(get_engine()) as session:
            row = session.get(Attempt, attempt_id)
            if row is None:
                return None
            events = session.scalars(
                select(LearningEvent)
                .where(LearningEvent.attempt_id == attempt_id)
                .order_by(LearningEvent.id)
            ).all()
            hints = session.scalars(
                select(HintRecord)
                .where(HintRecord.attempt_id == attempt_id)
                .order_by(HintRecord.id)
            ).all()
            return {
                "attempt_id": row.id,
                "student_id": row.student_id,
                "problem_id": row.problem_id,
                # ---- P11：Attempt 状态投影（服务重启后据此继续作答）----
                "status": row.status or STATUS_IN_PROGRESS,
                "current_node": row.current_node or "",
                "current_method": row.current_method or "",
                "visited_nodes": list(row.visited_nodes or []),
                "mistakes": list(row.mistakes or []),
                "missing_reasoning": list(row.missing_reasoning or []),
                "weak_concepts": list(row.weak_concepts or []),
                "stuck_score": float(row.stuck_score or 0.0),
                "hint_history": list(row.hint_history or []),
                "error_count": int(row.error_count or 0),
                "state": row.state,
                "start_time": row.start_time.isoformat() if row.start_time else None,
                "end_time": row.end_time.isoformat() if row.end_time else None,
                "final_correct": row.final_correct,
                "hint_count": row.hint_count,
                "stuck_count": row.stuck_count,
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "content": event.content,
                    }
                    for event in events
                ],
                "hints": [
                    {
                        "hint_level": hint.hint_level,
                        "hint": hint.hint,
                        "issue_type": hint.issue_type,
                    }
                    for hint in hints
                ],
            }
    except Exception as exc:  # noqa: BLE001
        _warn_once("db", exc)
        raise PersistenceUnavailable(str(exc)) from exc


@_db_guarded(raises=False)
def supersede_media_hints(attempt_id: str, source_id: str, revision: int) -> int:
    """Retain delivered hints, excluding replaced OCR evidence from effect metrics."""
    from datetime import datetime, timezone
    try:
        with Session(get_engine()) as session, session.begin():
            rows = session.execute(select(HintRecord).where(HintRecord.attempt_id == attempt_id)).scalars().all()
            count = 0
            for row in rows:
                meta = row.state_before or {}
                if meta.get("source_id") == source_id and meta.get("source_revision") == revision:
                    row.effective = None
                    row.effective_evidence = f"superseded OCR source {source_id} revision {revision}"
                    row.resolved_at = datetime.now(timezone.utc)
                    count += 1
            return count
    except Exception as exc:
        _warn_once("db", exc)
        return 0
