"""持久化层测试（Phase 6）。

- 降级行为：DB 不可用时写入返回 False、查询抛 PersistenceUnavailable（离线可测）
- 集成：PostgreSQL 可达时做真实读写；不可达则整类跳过
"""
import uuid

import pytest
from db_utils import db_available
from sqlalchemy import text

from app.db import persistence
from app.models.problem import Problem as ProblemSchema
from app.models.student import TimelineEvent


# ---------- 熔断器（P20 补漏） ----------

@pytest.fixture
def breaker_reset():
    """每个用例前后都清掉熔断状态，避免互相影响。"""
    persistence._breaker_open_until = 0.0
    yield
    persistence._breaker_open_until = 0.0


def test_breaker_stops_paying_the_timeout_on_every_call(monkeypatch, breaker_reset):
    """数据库不可用时，**只有第一次**付出连接超时，之后走熔断。

    不熔断的后果不是"降级"，而是"变成每个请求白等 6 秒的慢服务"：
    实测 postgres_connect_timeout=3s，而 localhost 同时解析到 ::1 与 127.0.0.1，
    psycopg 两个都试 → 单次 6 秒；一轮 /attempt/analyze 要写库多次。
    """
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)

    assert persistence.persist_trace("a", 1, {}) is False      # 第一次：真去连，失败
    assert persistence.breaker_status()["open"] is True

    # 之后不再触碰数据库：把 get_engine 换成"一碰就炸"，熔断期内应当根本不会调到它
    def explode():
        raise AssertionError("熔断期内不该再尝试连接数据库")

    monkeypatch.setattr(persistence, "get_engine", explode)
    assert persistence.persist_trace("a", 1, {}) is False
    assert persistence.persist_events("a", [TimelineEvent(event_type="write")]) is False


def test_breaker_reads_raise_while_writes_return_false(monkeypatch, breaker_reset):
    """读操作抛领域异常（调用方要能区分"没有"与"查不到"），写操作静默降级。"""
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)

    with pytest.raises(persistence.PersistenceUnavailable):
        persistence.get_attempt_row("a")        # 第一次：真去连，失败并熔断

    # 熔断期内读操作仍然抛同样的领域异常（调用方无需区分这两种情况）
    with pytest.raises(persistence.PersistenceUnavailable, match="熔断中"):
        persistence.get_attempt_row("a")
    assert persistence.persist_trace("a", 1, {}) is False


def test_breaker_lets_through_again_after_cooldown(monkeypatch, breaker_reset):
    """冷却结束后要重新放行——否则数据库恢复了后端也不会自愈。"""
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)
    persistence.persist_trace("a", 1, {})
    assert persistence.breaker_status()["open"] is True

    # 手动把冷却拨到过去，模拟"10 秒过去了"
    persistence._breaker_open_until = 0.0
    calls = []

    def counting_engine():
        calls.append(1)
        raise RuntimeError("还是连不上")

    monkeypatch.setattr(persistence, "get_engine", counting_engine)
    persistence.persist_trace("a", 1, {})

    assert calls, "冷却结束后应当重新尝试连接，而不是永久熔断"


def test_noop_calls_bypass_the_breaker(monkeypatch, breaker_reset):
    """空事件列表是 no-op，根本不需要数据库——不该被熔断影响。

    （这是接熔断器时踩到的回归：原本 persist_events(id, []) 返回 True，
    加了闸门之后变成 False，把"什么都没做"误报成了"降级失败"。）
    """
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)
    persistence.persist_trace("a", 1, {})       # 先熔断
    assert persistence.breaker_status()["open"] is True

    def explode():
        raise AssertionError("no-op 调用不该触碰数据库")

    monkeypatch.setattr(persistence, "get_engine", explode)
    assert persistence.persist_events("a", []) is True


# ---------- 降级行为（离线） ----------

def _broken_engine():
    raise RuntimeError("connection refused")


def test_writes_degrade_gracefully_when_db_down(monkeypatch):
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)
    assert persistence.persist_attempt_start("s", "p", "a") is False
    assert persistence.persist_events("a", [TimelineEvent(content="x")]) is False
    assert persistence.persist_hint("a", 1, "h", "concept") is False
    assert persistence.persist_attempt_finish("a", True, 1, 0) is False
    assert persistence.persist_profile_weak_concepts("s", ["概念"]) is False
    assert persistence.persist_trace("a", 1, {"step_match": {}}) is False


def test_get_traces_raises_when_db_down(monkeypatch):
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)
    with pytest.raises(persistence.PersistenceUnavailable):
        persistence.get_traces("a")


def test_get_attempt_detail_raises_when_db_down(monkeypatch):
    monkeypatch.setattr(persistence, "get_engine", _broken_engine)
    with pytest.raises(persistence.PersistenceUnavailable):
        persistence.get_attempt_detail("a")


def test_persist_events_empty_list_is_noop():
    assert persistence.persist_events("a", []) is True


# ---------- 集成（PostgreSQL 可达时执行） ----------

@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
@pytest.mark.real_db
class TestPersistenceIntegration:
    @pytest.fixture
    def ids(self):
        run = uuid.uuid4().hex[:8]
        data = {
            "student": f"stu_test_{run}",
            "problem": f"prob_test_{run}",
            "attempt": f"att{run}",
        }
        yield data
        with __import__("app.db.postgres", fromlist=["get_engine"]).get_engine().begin() as conn:
            conn.execute(
                text("DELETE FROM learning_events WHERE attempt_id=:a"), {"a": data["attempt"]}
            )
            conn.execute(
                text("DELETE FROM hint_records WHERE attempt_id=:a"), {"a": data["attempt"]}
            )
            conn.execute(text("DELETE FROM attempts WHERE id=:a"), {"a": data["attempt"]})
            conn.execute(
                text("DELETE FROM student_profiles WHERE student_id=:s"), {"s": data["student"]}
            )
            conn.execute(text("DELETE FROM students WHERE id=:s"), {"s": data["student"]})
            conn.execute(text("DELETE FROM problems WHERE id=:p"), {"p": data["problem"]})

    def test_full_roundtrip(self, ids):
        problem = ProblemSchema(problem_id=ids["problem"], title="T", content="C", answer="A")
        assert persistence.persist_attempt_start(
            ids["student"], ids["problem"], ids["attempt"], problem=problem
        )
        assert persistence.persist_events(
            ids["attempt"],
            [
                TimelineEvent(timestamp=1.0, event_type="submit", content="第一答"),
                TimelineEvent(timestamp=2.5, event_type="hint_request", content="请求提示"),
            ],
        )
        assert persistence.persist_hint(ids["attempt"], 1, "一级提示", "concept")
        assert persistence.persist_hint(ids["attempt"], 2, "二级提示", "concept")
        assert persistence.persist_attempt_finish(
            ids["attempt"], final_correct=True, hint_count=2, stuck_count=1
        )
        assert persistence.persist_profile_weak_concepts(ids["student"], ["机械能守恒条件"])
        assert persistence.persist_profile_weak_concepts(
            ids["student"], ["机械能守恒条件", "摩擦力做功"]
        )

        detail = persistence.get_attempt_detail(ids["attempt"])
        assert detail is not None
        assert detail["student_id"] == ids["student"]
        assert detail["problem_id"] == ids["problem"]
        assert [e["event_type"] for e in detail["events"]] == ["submit", "hint_request"]
        assert len(detail["hints"]) == 2
        assert detail["final_correct"] is True
        assert detail["end_time"] is not None
        assert detail["hint_count"] == 2
        assert detail["stuck_count"] == 1

        from sqlalchemy.orm import Session as SASession

        from app.db.postgres import get_engine
        from app.db.tables import StudentProfile

        with SASession(get_engine()) as session:
            profile = session.get(StudentProfile, ids["student"])
            # 画像取并集且保序
            assert profile.weak_concepts == ["机械能守恒条件", "摩擦力做功"]

    def test_missing_attempt_returns_none(self, ids):
        assert persistence.get_attempt_detail("no_such_attempt_xyz") is None

    def test_finish_nonexistent_attempt_returns_false(self, ids):
        assert persistence.persist_attempt_finish("ghost_attempt", True, 0, 0) is False

    def test_trace_roundtrip(self, ids):
        """P9 Tracing：按轮写入、按序读回。"""
        from app.models.problem import Problem as ProblemSchema

        problem = ProblemSchema(problem_id=ids["problem"], title="T", content="C")
        assert persistence.persist_attempt_start(
            ids["student"], ids["problem"], ids["attempt"], problem=problem
        )
        for rnd in range(1, 4):
            assert persistence.persist_trace(
                ids["attempt"], round_index=rnd,
                payload={"student_input": f"第{rnd}轮作答", "hint_level": rnd,
                         "step_match": {"matched_nodes": [f"S{rnd}"]}},
            )
        traces = persistence.get_traces(ids["attempt"])
        assert [t["round_index"] for t in traces] == [1, 2, 3]
        assert traces[0]["payload"]["student_input"] == "第1轮作答"
        assert traces[2]["payload"]["step_match"]["matched_nodes"] == ["S3"]
