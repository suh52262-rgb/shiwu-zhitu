"""Attempt 生命周期与 StudentState 持久化测试（P11）。

覆盖任务书 P11 的五个接口与验收标准：

    连续进行 10 轮学生输入后 → 服务重启 → 再加载 attempt →
    current_node / visited_nodes / mistakes / hint_history 等状态仍然存在 →
    Agent 能从之前状态继续工作。

其中"服务重启"用**新建一个 AttemptStore 实例**模拟（内存为空，只能从数据库读回），
因此该用例标了 `real_db`；其余用例走 conftest 提供的内存降级路径，完全离线。
"""
import logging
import uuid

import pytest
from db_utils import db_available
from sqlalchemy import text

from app.db import persistence
from app.db.migrate import pending_columns, pending_indexes
from app.models.attempt import (
    STATUS_ABANDONED,
    STATUS_FINISHED,
    STATUS_IN_PROGRESS,
    AttemptRecord,
)
from app.models.student import StudentState
from app.services.attempt_store import AttemptStore


@pytest.fixture
def store(isolated_attempt_store):
    """本用例独占的 AttemptStore（默认纯内存模式，见 conftest）。"""
    return isolated_attempt_store


def _round(state: StudentState, node: str, *, error: bool = False, hint: str = "") -> StudentState:
    """模拟一轮 Agent 运行对 StudentState 的推进（不依赖 LLM）。"""
    state.current_node = node
    state.visited_nodes = [*state.visited_nodes, node]
    state.current_method = "能量法"
    state.has_obvious_error = error
    if error:
        state.diagnosed_issue = f"节点 {node} 推理缺失"
        state.issue_type = "concept"
        state.severity = "medium"
        state.missing_reasoning = ["缺少受力分析"]
    if hint:
        state.hint = hint
    return state


# ---------- 五个生命周期接口 ----------

def test_create_attempt_returns_open_record_with_initial_state(store):
    record = store.create_attempt("stu_a", "phy_001")

    assert record.attempt_id and len(record.attempt_id) == 12
    assert record.status == STATUS_IN_PROGRESS
    assert record.is_open is True
    assert record.visited_nodes == []
    assert record.mistakes == []
    assert record.hint_history == []
    assert record.started_at is not None
    assert isinstance(record.state, StudentState)
    assert record.state.student_id == "stu_a"
    assert record.state.problem_id == "phy_001"


def test_load_attempt_roundtrip(store):
    created = store.create_attempt("stu_a", "phy_001")
    loaded = store.load_attempt(created.attempt_id)

    assert loaded is not None
    assert loaded.attempt_id == created.attempt_id
    assert loaded.student_id == "stu_a"


def test_load_attempt_returns_none_for_unknown_id(store):
    assert store.load_attempt("no_such_attempt") is None


def test_save_attempt_writes_projection_and_snapshot(store):
    record = store.create_attempt("stu_a", "phy_001")
    state = _round(record.state, "S1")

    updated = store.save_attempt(record, state, error_count=0, hint_count=1, stuck_count=0)

    assert updated.current_node == "S1"
    assert updated.visited_nodes == ["S1"]
    assert updated.current_method == "能量法"
    assert updated.hint_count == 1
    assert updated.state.current_node == "S1"

    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.current_node == "S1"
    assert reloaded.visited_nodes == ["S1"]
    assert reloaded.state.visited_nodes == ["S1"]


def test_save_attempt_records_mistakes_once_per_node_issue(store):
    record = store.create_attempt("stu_a", "phy_002")
    for _ in range(3):  # 同一节点同一问题重复三轮
        record = store.save_attempt(record, _round(record.state, "S10", error=True))

    assert len(record.mistakes) == 1
    assert record.mistakes[0]["node"] == "S10"
    assert record.mistakes[0]["issue_type"] == "concept"


def test_update_attempt_merges_partial_fields(store):
    record = store.create_attempt("stu_a", "phy_001")
    ok = store.update_attempt(
        record.attempt_id,
        current_node="S3",
        hint_history=[{"hint_level": 1, "hint": "想一想", "node": "S3"}],
    )

    assert ok is True
    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.current_node == "S3"
    assert reloaded.hint_history[0]["hint_level"] == 1
    assert reloaded.status == STATUS_IN_PROGRESS  # 未被误改


def test_update_attempt_ignores_unknown_fields(store):
    """未知字段被忽略，但**同一调用里的合法字段仍要生效**。

    原来只断言 `not hasattr(reloaded, "not_a_field")` —— Pydantic 默认忽略多余字段，
    这条断言恒真：把白名单整个删掉也照样绿。这里补上"合法字段确实写入"这一侧。
    """
    record = store.create_attempt("stu_a", "phy_001")
    store.update_attempt(record.attempt_id, not_a_field="x", current_node="S2")

    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.current_node == "S2"
    assert "not_a_field" not in AttemptRecord.model_fields


def test_mutable_field_whitelist_excludes_identity_and_creation_time():
    """白名单不得允许改主键与创建时间（生命周期字段由 finish/abandon 负责）。"""
    assert "attempt_id" not in persistence.ATTEMPT_MUTABLE_FIELDS
    assert "started_at" not in persistence.ATTEMPT_MUTABLE_FIELDS
    assert "student_id" not in persistence.ATTEMPT_MUTABLE_FIELDS


def test_finish_attempt_sets_terminal_status(store):
    record = store.create_attempt("stu_a", "phy_001")
    assert store.finish_attempt(record.attempt_id, final_correct=True) is True

    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.status == STATUS_FINISHED
    assert reloaded.final_correct is True
    assert reloaded.finished_at is not None
    assert reloaded.is_open is False


def test_abandon_attempt_keeps_history(store):
    record = store.create_attempt("stu_a", "phy_001")
    store.abandon_attempt(record.attempt_id)

    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.status == STATUS_ABANDONED
    assert reloaded.finished_at is not None  # 放弃也要收尾


def test_finish_attempt_returns_false_for_unknown_id(store):
    assert store.finish_attempt("no_such_attempt") is False


# ---------- 未结束 attempt 的续做 ----------

def test_load_open_attempt_finds_unfinished(store):
    record = store.create_attempt("stu_a", "phy_001")
    store.save_attempt(record, _round(record.state, "S1"))

    reopened = store.load_open_attempt("stu_a", "phy_001")
    assert reopened is not None
    assert reopened.attempt_id == record.attempt_id
    assert reopened.visited_nodes == ["S1"]


def test_load_open_attempt_skips_finished(store):
    record = store.create_attempt("stu_a", "phy_001")
    store.finish_attempt(record.attempt_id, final_correct=True)

    assert store.load_open_attempt("stu_a", "phy_001") is None


def test_load_open_attempt_is_scoped_by_student_and_problem(store):
    store.create_attempt("stu_a", "phy_001")
    assert store.load_open_attempt("stu_b", "phy_001") is None
    assert store.load_open_attempt("stu_a", "phy_002") is None


def test_new_attempt_after_finish_is_independent(store):
    first = store.create_attempt("stu_a", "phy_001")
    store.save_attempt(first, _round(first.state, "S1"))
    store.finish_attempt(first.attempt_id, final_correct=True)

    second = store.create_attempt("stu_a", "phy_001")
    assert second.attempt_id != first.attempt_id
    assert second.visited_nodes == []          # 新作答从零开始
    assert store.load_attempt(first.attempt_id).visited_nodes == ["S1"]  # 历史保留


# ---------- 降级：数据库不可用时仍保持多轮连续 ----------

def test_memory_fallback_keeps_multi_turn_continuity(store):
    """纯内存模式（conftest 给的 persistent=False）下会话跨轮连续。"""
    record = store.create_attempt("stu_a", "phy_001")
    state = record.state
    for node in ("S1", "S2", "S3"):
        state = _round(state, node)
        record = store.save_attempt(record, state)

    resumed = store.load_open_attempt("stu_a", "phy_001")
    assert resumed.visited_nodes == ["S1", "S2", "S3"]
    assert resumed.current_node == "S3"
    assert resumed.state.visited_nodes == ["S1", "S2", "S3"]


def test_real_degradation_path_when_db_is_down(monkeypatch):
    """真正的降级路径：`persistent=True` 但数据库不可用 → 退回内存存储。

    与上面那条不是一回事：`persistent=False` 是"从不碰数据库"的纯内存模式，
    走不到生产代码里任何一条降级分支。这里把持久化层打挂，逼它走
    persist_* 返回 False / get_* 抛 PersistenceUnavailable 的真实分支。
    """
    def unavailable(*args, **kwargs):
        raise persistence.PersistenceUnavailable("模拟数据库不可用")

    monkeypatch.setattr(persistence, "get_attempt_row", unavailable)
    monkeypatch.setattr(persistence, "get_open_attempt_row", unavailable)
    monkeypatch.setattr(persistence, "persist_attempt_start", lambda *a, **k: False)
    monkeypatch.setattr(persistence, "save_attempt_state", lambda *a, **k: False)
    monkeypatch.setattr(persistence, "finish_attempt", lambda *a, **k: False)

    store = AttemptStore(persistent=True)      # 注意：不是 conftest 的纯内存实例

    record = store.create_attempt("stu_deg", "phy_001")
    state = record.state
    for node in ("S1", "S2"):
        state = _round(state, node)
        record = store.save_attempt(record, state, error_count=1, hint_count=1)

    reloaded = store.load_attempt(record.attempt_id)
    assert reloaded.visited_nodes == ["S1", "S2"]
    assert reloaded.current_node == "S2"
    assert reloaded.error_count == 1           # 计数在降级路径上也要生效

    assert store.load_open_attempt("stu_deg", "phy_001") is not None
    assert store.update_attempt(record.attempt_id, current_node="S3") is True
    assert store.finish_attempt(record.attempt_id, final_correct=True) is True
    assert store.load_attempt(record.attempt_id).status == STATUS_FINISHED


# ---------- 内存降级存储的单元行为 ----------

def test_clear_memory_drops_records():
    store = AttemptStore(persistent=False)   # 纯内存模式，才能观察到清空效果
    record = store.create_attempt("stu_x", "phy_001")
    assert store.load_attempt(record.attempt_id) is not None

    store.clear_memory()
    assert store.load_attempt(record.attempt_id) is None


def test_corrupted_state_snapshot_falls_back_to_empty_state(caplog):
    """快照无法还原时必须退回空状态，而不是让整个会话不可用。

    断言要点：光看 student_id / problem_id 是不够的——**正常解析与降级都会带上它们**，
    那种断言恒为真、根本区分不出走了哪条分支。这里改为检查"降级日志确实打了"
    且"状态确实被重置"。
    """
    from app.services.attempt_store import _to_student_state

    with caplog.at_level(logging.WARNING, logger="app.services.attempt_store"):
        state = _to_student_state({"current_node": 123}, "stu", "phy")

    assert any("无法还原" in record.message for record in caplog.records), "未走降级分支"
    assert state.current_node == ""          # 被重置，而不是 123 或 "123"
    assert state.student_id == "stu"
    assert state.problem_id == "phy"


def test_valid_state_snapshot_is_preserved():
    """对照组：合法快照必须被完整还原——否则上面那条用例也可能是恒真的。"""
    from app.services.attempt_store import _to_student_state

    state = _to_student_state(
        {"current_node": "S5", "visited_nodes": ["S1", "S5"], "hint_level": 2},
        "stu",
        "phy",
    )

    assert state.current_node == "S5"
    assert state.visited_nodes == ["S1", "S5"]
    assert state.hint_level == 2


# ---------- P11 验收：10 轮后"重启服务" ----------

@pytest.mark.real_db
@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
def test_acceptance_state_survives_service_restart():
    """P11 验收标准：10 轮输入 → 重启 → 状态仍在 → Agent 能从原状态继续。"""
    student = f"stu_p11_{uuid.uuid4().hex[:8]}"
    from app.models.problem import Problem as ProblemSchema
    problem_id = "p11_restart_" + uuid.uuid4().hex[:12]
    problem = ProblemSchema(problem_id=problem_id, title="Synthetic restart fixture", content="Test only")
    nodes = ["S1", "S2", "S3", "S4", "S5", "S6a", "S7", "S8", "S9", "S10"]

    # --- 作答进程：连续 10 轮 ---
    writer = AttemptStore()
    record = writer.create_attempt(student, problem_id, problem=problem)
    for index, node in enumerate(nodes):
        state = _round(record.state, node, error=(index == 4))
        record = writer.save_attempt(record, state)
        writer.update_attempt(
            record.attempt_id,
            hint_history=[
                *record.hint_history,
                {"hint_level": 1 + index % 3, "hint": f"第 {index + 1} 轮提示", "node": node},
            ],
        )
        record = writer.load_attempt(record.attempt_id)

    attempt_id = record.attempt_id
    try:
        # --- 模拟服务重启：全新 store，内存为空，只能从数据库读回 ---
        restarted = AttemptStore()
        loaded = restarted.load_attempt(attempt_id)

        assert loaded is not None
        assert loaded.status == STATUS_IN_PROGRESS
        assert loaded.current_node == "S10"
        assert loaded.visited_nodes == nodes
        assert loaded.current_method == "能量法"
        assert len(loaded.mistakes) == 1 and loaded.mistakes[0]["node"] == "S5"
        assert len(loaded.hint_history) == 10
        assert loaded.hint_history[-1]["node"] == "S10"
        assert loaded.state.visited_nodes == nodes
        assert loaded.state.timeline == []

        # --- Agent 能从之前状态继续：续做同一个 attempt ---
        resumed = restarted.load_open_attempt(student, problem_id)
        assert resumed is not None
        assert resumed.attempt_id == attempt_id

        continued = restarted.save_attempt(resumed, _round(resumed.state, "S11"))
        assert continued.visited_nodes == [*nodes, "S11"]
        assert continued.current_node == "S11"

        final = restarted.finish_attempt(attempt_id, final_correct=True)
        assert final is True
        assert restarted.load_attempt(attempt_id).status == STATUS_FINISHED
    finally:
        _purge_student(student)
        from app.db.postgres import get_engine
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM problems WHERE id=:p"), {"p": problem_id})


@pytest.mark.real_db
@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
def test_database_schema_matches_orm():
    """ORM 加了列/索引但忘记跑迁移时，这条用例应当失败（守卫 scripts/init_db.py）。

    索引这一项必须一起守：ADD COLUMN 不会带上 index=True 声明的索引，
    否则"新建的库有索引、迁移过的库没有"，两个环境悄悄跑在不同结构上。
    """
    assert pending_columns() == {}
    assert pending_indexes() == {}


@pytest.mark.real_db
@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
def test_attempt_detail_exposes_state_projection():
    student = f"stu_p11d_{uuid.uuid4().hex[:8]}"
    store = AttemptStore()
    record = store.create_attempt(student, "phy_001")
    record = store.save_attempt(record, _round(record.state, "S1", error=True))
    try:
        detail = persistence.get_attempt_detail(record.attempt_id)
        assert detail["status"] == STATUS_IN_PROGRESS
        assert detail["current_node"] == "S1"
        assert detail["visited_nodes"] == ["S1"]
        assert detail["mistakes"][0]["node"] == "S1"
        assert detail["state"]["current_node"] == "S1"
    finally:
        _purge_student(student)


def _purge_student(student_id: str) -> None:
    """清理集成测试数据（events/hints/traces 随 attempt 级联删除）。"""
    from app.db.postgres import get_engine

    try:
        with get_engine().begin() as conn:
            conn.execute(text("DELETE FROM attempts WHERE student_id=:s"), {"s": student_id})
            conn.execute(
                text("DELETE FROM student_profiles WHERE student_id=:s"), {"s": student_id}
            )
            conn.execute(text("DELETE FROM students WHERE id=:s"), {"s": student_id})
    except Exception:  # noqa: BLE001 清理失败不影响断言结论
        pass
