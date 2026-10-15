"""Attempt 生命周期与 StudentState 持久化（P11）。

任务书《P11-P25 平台化任务书》（docs/specs/03-第三阶段-P11-P25-平台化.md）第 P11 节要求：

    create_attempt() / load_attempt() / save_attempt() / update_attempt() / finish_attempt()
    LangGraph 每次运行前 load StudentState，运行完成后 save StudentState；
    不要依赖 LLM 自己记住之前发生了什么。

设计要点
--------
1. **数据库为准**：PostgreSQL 可达时，Attempt 与 StudentState 全部落库，
   因此"服务重启 → 重新加载 attempt → 状态仍在"成立（P11 验收标准）。
2. **降级不降功能**：数据库不可用时，同一个 AttemptStore 退回进程内存储，
   多轮会话仍然连续（延续项目既有的"无 DB 也能跑"承诺）。两者共用同一套
   AttemptRecord 结构与投影逻辑，行为一致。
3. **状态投影与快照同写**：`_project()` 是唯一从 StudentState 推导投影字段的地方，
   避免投影与快照漂移。

注意（沿用接手时的风险记录）：内存降级存储是**单进程**的，多 worker 部署下
同一次作答会分裂；数据库可用时不存在该问题。
"""
import logging
import threading
import uuid
from datetime import datetime, timezone

from app.db import persistence
from app.models.attempt import (
    STATUS_ABANDONED,
    STATUS_FINISHED,
    STATUS_IN_PROGRESS,
    AttemptRecord,
)
from app.models.problem import Problem as ProblemSchema
from app.models.student import StudentState

logger = logging.getLogger(__name__)


def new_attempt_id() -> str:
    """attempt_id：与既有格式保持一致（uuid4 前 12 位十六进制）。"""
    return uuid.uuid4().hex[:12]


def _to_student_state(raw: dict | None, student_id: str, problem_id: str) -> StudentState:
    """把落库的 state JSON 还原成 StudentState；缺失/损坏时退回空状态。"""
    if not raw:
        return StudentState(student_id=student_id, problem_id=problem_id)
    try:
        state = StudentState(**raw)
    except Exception as exc:  # noqa: BLE001 快照结构变更时不能让会话彻底不可用
        logger.warning("StudentState 快照无法还原，退回空状态：%s", exc)
        return StudentState(student_id=student_id, problem_id=problem_id)
    state.student_id = student_id or state.student_id
    state.problem_id = problem_id or state.problem_id
    return state


def _project(state: StudentState, record: AttemptRecord) -> dict:
    """由 StudentState 推导 Attempt 的状态投影（唯一入口，防漂移）。

    `mistakes` 与 `hint_history` 是累积量，由调用方通过 append 维护，这里只
    补齐能从状态直接读出的部分。
    """
    mistakes = list(record.mistakes)
    if state.has_obvious_error and state.diagnosed_issue:
        entry = {
            "node": state.current_node,
            "issue_type": state.issue_type,
            "issue": state.diagnosed_issue,
            "severity": state.severity,
        }
        # 同一节点同一问题只记一次，避免重复轮次把列表刷爆
        if not any(
            item.get("node") == entry["node"]
            and item.get("issue") == entry["issue"]
            for item in mistakes
        ):
            mistakes.append(entry)

    return {
        "current_node": state.current_node,
        "current_method": state.current_method,
        "visited_nodes": list(state.visited_nodes),
        "mistakes": mistakes,
        "missing_reasoning": list(state.missing_reasoning),
        "weak_concepts": list(state.weak_concepts),
        "stuck_score": float(state.stuck_score or 0.0),
    }


class AttemptStore:
    """Attempt 生命周期门面：PostgreSQL 优先，不可用时退化为进程内存储。

    `persistent=False` 表示**纯内存模式**：完全不触碰数据库。用于
    ① 单测离线运行；② 需要确定性地验证降级路径。
    """

    def __init__(self, persistent: bool = True) -> None:
        self.persistent = persistent
        self._memory: dict[str, AttemptRecord] = {}
        self._locks: dict[tuple[str, str], threading.RLock] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------- 请求级互斥

    def attempt_lock(self, student_id: str, problem_id: str) -> threading.RLock:
        """取得"该学生该题"的互斥锁 —— 一次作答的处理必须串行。

        为什么需要它：一次 `/attempt/analyze` 的读-改-写跨越**整个请求**
        （load 状态 → 跑 LLM 工作流（秒级）→ 写回），单次操作内部的锁保护不到这么长的窗口。
        并发的两个请求会各自基于同一份旧快照写回，导致：
        visited_nodes 回退、hint_history 丢条目；
        更糟的是 `load_open_attempt` → `create_attempt` 非原子，会造出两个
        `in_progress` 的 attempt（幽灵会话）。

        语义上也应该串行：同一个学生本来就是一题一步地作答。

        局限：只能约束**单进程**。多 worker 部署需要数据库层乐观锁
        （见 `AttemptStore` 模块 docstring 的说明）。
        锁字典按 (student_id, problem_id) 增长，量级等同"学生数 × 题数"，可接受。
        """
        key = (student_id, problem_id)
        with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    # ---------------------------------------------------------- 生命周期

    def create_attempt(
        self,
        student_id: str,
        problem_id: str,
        problem: ProblemSchema | None = None,
        attempt_id: str | None = None,
    ) -> AttemptRecord:
        """开始一次作答：落库（或在内存中）建立 attempt 与本轮初始状态。"""
        attempt_id = attempt_id or new_attempt_id()
        record = AttemptRecord(
            attempt_id=attempt_id,
            student_id=student_id,
            problem_id=problem_id,
            status=STATUS_IN_PROGRESS,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            state=StudentState(
                student_id=student_id,
                problem_id=problem_id,
                problem_content=problem.full_text if problem else "",
                graph_binding_status="unbound",
            ),
        )
        saved = self.persistent and persistence.persist_attempt_start(
            student_id, problem_id, attempt_id, problem
        )
        if saved:
            persisted = self._read_db(attempt_id)
            if persisted is not None:
                persisted.state = record.state
                self.save_attempt(persisted, record.state)
                # 以库中行为准（含 start_time 等由数据库生成的字段）
                return persisted
        with self._lock:
            self._memory[attempt_id] = record
        # 返回深拷贝：否则调用方拿到的是 store 内部那个活对象，
        # 对它（尤其是它内部的 StudentState）的任何改写都会直接落到存储里，
        # 与 load_attempt 的"返回副本"语义不一致，出错时内存会留下半成品状态。
        return record.model_copy(deep=True)

    def load_attempt(self, attempt_id: str) -> AttemptRecord | None:
        """按 id 加载 attempt；不存在返回 None。"""
        record = self._read_db(attempt_id)
        if record is not None:
            return record
        with self._lock:
            record = self._memory.get(attempt_id)
        return record.model_copy(deep=True) if record else None

    def load_open_attempt(self, student_id: str, problem_id: str) -> AttemptRecord | None:
        """加载该学生该题最近一次未结束的作答（服务重启后续做入口）。"""
        row = None
        if self.persistent:
            try:
                row = persistence.get_open_attempt_row(student_id, problem_id)
            except persistence.PersistenceUnavailable:
                row = None
        if row is not None:
            return AttemptRecord(
                **{**row, "state": _to_student_state(row["state"], student_id, problem_id)}
            )
        # 内存降级：取该学生该题最后一个仍 in_progress 的记录
        with self._lock:
            candidates = [
                item
                for item in self._memory.values()
                if item.student_id == student_id
                and item.problem_id == problem_id
                and item.status == STATUS_IN_PROGRESS
            ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda item: item.updated_at or datetime.min.replace(tzinfo=timezone.utc))
        return latest.model_copy(deep=True)

    def save_attempt(
        self,
        record: AttemptRecord,
        state: StudentState,
        *,
        error_count: int | None = None,
        hint_count: int | None = None,
        stuck_count: int | None = None,
    ) -> AttemptRecord:
        """把本轮结束后的 StudentState 写回 attempt（投影 + 完整快照 + 计数）。"""
        updated = record.model_copy(deep=True)
        updated.state = state
        updated.error_count = record.error_count if error_count is None else error_count
        updated.hint_count = record.hint_count if hint_count is None else hint_count
        updated.stuck_count = record.stuck_count if stuck_count is None else stuck_count
        updated.updated_at = datetime.now(timezone.utc)
        for key, value in _project(state, updated).items():
            setattr(updated, key, value)

        payload = updated.model_dump(mode="json", exclude={"attempt_id", "student_id", "problem_id"})
        # 时间与**生命周期字段**一律不由这里写：
        # started_at 由数据库/创建时确定；
        # status / final_correct 只由 finish_attempt / abandon_attempt 决定。
        # 否则一个迟到的 save（快照里 status 还是 in_progress）会把已经结束的
        # attempt 打回 in_progress、把 final_correct 清成 NULL，
        # 之后 load_open_attempt 还会把这个"已结束"的 attempt 重新挑出来续做。
        for key in ("started_at", "updated_at", "finished_at", "status", "final_correct"):
            payload.pop(key, None)
        self.update_attempt(updated.attempt_id, **payload)
        return updated

    def update_attempt(self, attempt_id: str, **fields: object) -> bool:
        """部分更新 attempt（同时写库与内存镜像，保证降级时状态不丢）。

        返回 True 表示改动已被**当前生效的后端**接受：数据库写入成功，
        或数据库不可用但内存降级存储已更新。
        """
        applied = False
        with self._lock:
            current = self._memory.get(attempt_id)
            if current is not None:
                data = current.model_dump()
                data.update({k: v for k, v in fields.items() if k in AttemptRecord.model_fields})
                self._memory[attempt_id] = AttemptRecord(**data)
                applied = True
        if not self.persistent:
            return applied
        try:
            return persistence.save_attempt_state(attempt_id, **fields) or applied
        except persistence.PersistenceUnavailable:
            return applied

    def finish_attempt(
        self,
        attempt_id: str,
        final_correct: bool | None = None,
        status: str = STATUS_FINISHED,
    ) -> bool:
        """结束作答（完成 / 放弃），写入终态与结束时间。

        返回值语义同 `update_attempt`：数据库或内存降级存储任一落实即为 True。
        """
        now = datetime.now(timezone.utc)
        applied = False
        with self._lock:
            current = self._memory.get(attempt_id)
            if current is not None:
                current.status = status
                current.finished_at = current.finished_at or now
                current.updated_at = now
                if final_correct is not None:
                    current.final_correct = final_correct
                applied = True
        if not self.persistent:
            return applied
        try:
            return (
                persistence.finish_attempt(
                    attempt_id, final_correct=final_correct, status=status
                )
                or applied
            )
        except persistence.PersistenceUnavailable:
            return applied

    def abandon_attempt(self, attempt_id: str) -> bool:
        """放弃作答（保留历史，供"重新开始本题"使用）。"""
        return self.finish_attempt(attempt_id, final_correct=False, status=STATUS_ABANDONED)

    def clear_memory(self) -> None:
        """清空进程内降级存储（测试隔离与运维重置用；不影响数据库）。"""
        with self._lock:
            self._memory.clear()

    # ---------------------------------------------------------- 内部

    def _read_db(self, attempt_id: str) -> AttemptRecord | None:
        if not self.persistent:
            return None
        try:
            row = persistence.get_attempt_row(attempt_id)
        except persistence.PersistenceUnavailable:
            return None
        if row is None:
            return None
        return AttemptRecord(
            **{**row, "state": _to_student_state(row["state"], row["student_id"], row["problem_id"])}
        )


_STORE = AttemptStore()


def get_attempt_store() -> AttemptStore:
    """进程内单例（内存降级镜像需要跨请求共享）。"""
    return _STORE
