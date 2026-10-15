"""POST /attempt/analyze 接口测试（Phase 6 更新）。

- 假 LLM，验证会话状态跨请求累积、提示升级、attempt_id 与落库行为
- GET /attempt/{id} 的 503 / 200 / 404 分支
- PostgreSQL 可达时的真实落库集成测试（不可达则跳过）
"""
import uuid

import pytest
from db_utils import db_available
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.agent.step_matcher import StepMatch
from app.db import persistence
from app.main import app
from fakes import BAD_ANALYSIS, GOOD_ANALYSIS, install_fake_llm

client = TestClient(app)

PAYLOAD = {
    "student_id": "001",
    "problem_id": "phy_001",
    "answer": "mgh = 1/2mv^2，机械能守恒",
}


@pytest.fixture
def store(isolated_attempt_store):
    """本用例独占的 AttemptStore（默认纯内存模式，见 conftest）。"""
    return isolated_attempt_store


def _store_state(store, attempt_id: str):
    """取出某个 attempt 的 StudentState（内存或数据库后端都适用）。"""
    record = store.load_attempt(attempt_id)
    assert record is not None
    return record.state


def _post(**overrides) -> dict:
    return client.post("/attempt/analyze", json={**PAYLOAD, **overrides}).json()


def test_first_call_returns_level1_hint_and_attempt_id(monkeypatch):
    install_fake_llm(monkeypatch)
    body = _post()

    assert body["attempt_id"]  # Phase 6：每次会话有唯一 attempt_id
    assert body["hint_level"] == 1
    assert body["issue"] == "未考虑摩擦力做功"
    assert body["stuck_score"] == 0.0
    assert body["finished"] is False


def test_hint_escalates_while_error_persists(monkeypatch):
    install_fake_llm(monkeypatch)
    # P5 策略：连续失败轮数驱动级别（graphless：1→2→3→4，封顶 4）
    for expected in (1, 2, 3, 4, 4):
        assert _post()["hint_level"] == expected


def test_hint_policy_reasons_exposed(monkeypatch):
    install_fake_llm(monkeypatch)
    body = _post()
    assert body["hint_reasons"]  # 决策可解释


def test_request_hint_first_time_gives_concept_level(monkeypatch):
    """P5：首次主动请求 → 至少 L2 概念提示。"""
    install_fake_llm(monkeypatch)
    body = _post(request_hint=True)
    assert body["hint_level"] == 2
    assert any("主动请求" in r for r in body["hint_reasons"])


def test_blank_help_reuses_saved_ocr_and_escalates_without_resubmission(monkeypatch, store):
    install_fake_llm(monkeypatch,
                     step_match=StepMatch(matched_nodes=["S5"], confidence=0.95,
                                          correct=True, missing_reasoning=["未展示分离条件"]))
    first = _post(problem_id="phy_002", answer="P、Q分离时 N=0", request_hint=True)
    assert first["hint_level"] == 2 and not first["finished"]
    attempt_id = first["attempt_id"]
    for level in (3, 4, 5):
        response = client.post("/attempt/analyze", json={**PAYLOAD, "problem_id": "phy_002",
            "attempt_id": attempt_id, "answer": "", "request_hint": True})
        assert response.status_code == 200
        assert response.json()["hint_level"] == level
        state = _store_state(store, attempt_id)
        assert state.current_answer == "P、Q分离时 N=0"
        assert sum(event.event_type == "submit" for event in state.timeline) == 1
    assert sum(event.event_type == "hint_request" for event in state.timeline) == 4


def test_correct_graph_round_is_level0_no_hint(monkeypatch):
    """P5：本轮正确推进 → L0 不提示。"""
    install_fake_llm(
        monkeypatch,
        analysis=GOOD_ANALYSIS,
        step_match=StepMatch(matched_nodes=["S7"], confidence=0.95, correct=True),
    )
    body = _post(problem_id="phy_002", answer="v = d√(k/(m1+m2))")

    assert body["hint_level"] == 0
    assert body["hint"] == ""


def test_stuck_score_accumulates_with_errors(monkeypatch):
    install_fake_llm(monkeypatch)
    assert _post(idle_time=50)["stuck_score"] == pytest.approx(0.30)
    assert _post(idle_time=50)["stuck_score"] == pytest.approx(0.30)
    assert _post(idle_time=50)["stuck_score"] == pytest.approx(0.55)


def test_request_hint_adds_timeline_event_and_score(monkeypatch, store):
    install_fake_llm(monkeypatch)
    body = _post(request_hint=True)

    assert body["stuck_score"] == pytest.approx(0.20)
    types = [e.event_type for e in _store_state(store, body["attempt_id"]).timeline]
    assert types == ["submit", "hint_request"]


def test_finished_on_correct_answer(monkeypatch):
    install_fake_llm(monkeypatch, analysis=GOOD_ANALYSIS)
    body = _post(answer="W_f = 1/2mv^2 - mgh，机械能不守恒")

    assert body["finished"] is True
    assert body["issue_type"] == "none"


def test_reset_session_gives_new_attempt_id(monkeypatch):
    install_fake_llm(monkeypatch)
    first = _post()
    client.post("/attempt/reset", json={"student_id": "001", "problem_id": "phy_001"})
    second = _post()

    assert second["hint_level"] == 1
    assert second["attempt_id"] != first["attempt_id"]


def test_problem_content_loaded_from_store(monkeypatch, store):
    install_fake_llm(monkeypatch)
    body = _post()
    assert "机械能守恒" in _store_state(store, body["attempt_id"]).problem_content


def test_analyze_trace_records_prompt_versions(monkeypatch):
    """P13：每轮 Trace 记录本轮实际使用的提示词版本，供 A/B 对比与回归定位。"""
    install_fake_llm(monkeypatch)
    captured: list[dict] = []
    monkeypatch.setattr(
        persistence,
        "persist_trace",
        lambda *args, **kwargs: captured.append(kwargs.get("payload", {})) or True,
    )

    _post()

    assert captured, "本轮没有写入 trace"
    payload = captured[0]
    for role in ("step_matcher", "analyze_step", "diagnosis", "tutor"):
        value = payload.get(f"{role}_prompt_version")
        assert value and value.startswith("v"), f"缺少 {role}_prompt_version"


# ---------- P2：Problem Graph 模式 ----------

def test_graph_problem_returns_step_match_and_updates_visited(monkeypatch):
    install_fake_llm(monkeypatch)  # 默认 step_match = ["S5"]
    body = _post(problem_id="phy_002", answer="弹簧恢复原长时二者分离")

    assert body["graph_mode"] is True    # P8：显式 graph 模式
    assert body["step_match"] is not None
    assert body["step_match"]["matched_nodes"] == ["S5"]
    assert body["current_node"] == "S5"
    assert body["visited_nodes"] == ["S5"]
    # 跳步信息进入了诊断提示（missing_reasoning 非空）
    assert body["step_match"]["missing_reasoning"]


def test_visited_nodes_accumulate_across_rounds(monkeypatch):
    install_fake_llm(monkeypatch)
    _post(problem_id="phy_002", answer="第一轮：S5")
    install_fake_llm(
        monkeypatch,
        step_match=StepMatch(matched_nodes=["S3", "S4"], confidence=0.95, correct=True),
    )
    body = _post(problem_id="phy_002", answer="分离时 N=0，对 Q 有 N=m2a，故 a=0")

    assert body["step_match"]["matched_nodes"] == ["S3", "S4"]
    # 已访问节点按顺序累积且不重复
    assert body["visited_nodes"] == ["S5", "S3", "S4"]


def test_graphless_problem_has_no_step_match(monkeypatch):
    install_fake_llm(monkeypatch)
    body = _post()  # phy_001 没有 Problem Graph

    assert body["graph_mode"] is False   # P8：显式 graphless 模式
    assert body["step_match"] is None
    assert body["current_node"] == ""
    assert body["visited_nodes"] == []


# ---------- P4：图感知卡顿升级 ----------

def test_graph_aware_stuck_escalates_on_repeated_node_errors(monkeypatch):
    """在同一节点反复出错：图信号让卡顿分逐轮升高，并给出可读归因。"""
    install_fake_llm(
        monkeypatch,
        step_match=StepMatch(
            matched_nodes=[], attempted_nodes=["S10"], confidence=0.9, correct=False
        ),
    )
    scores = []
    last_reasons = []
    for _ in range(3):
        body = _post(problem_id="phy_002", answer="分离后振幅还是 d")
        scores.append(body["stuck_score"])
        last_reasons = body["stuck_reasons"]

    # 提示强度不再制造薄弱标签，因此没有旧政策附加的 0.10。
    # 仍保留重复错误产生的启发式分数；它不代表认知卡顿已验证。
    assert scores == pytest.approx([0.0, 0.0, 0.40])
    assert any("S10" in r for r in last_reasons)


def test_graphless_stuck_reasons_still_work(monkeypatch):
    install_fake_llm(monkeypatch)
    body = _post(idle_time=50)  # phy_001 无 Graph：只有行为信号
    assert body["stuck_reasons"] == ["空闲超过 40 秒"]
    assert body["stuck_score"] == pytest.approx(0.30)


# ---------- GET /attempt/{id} ----------

def test_get_attempt_503_when_db_down(monkeypatch):
    def _unavailable(attempt_id):
        raise persistence.PersistenceUnavailable("connection refused")

    monkeypatch.setattr(persistence, "get_attempt_detail", _unavailable)
    resp = client.get("/attempt/some_id")
    assert resp.status_code == 503


def test_get_attempt_404_when_missing(monkeypatch):
    monkeypatch.setattr(persistence, "get_attempt_detail", lambda attempt_id: None)
    resp = client.get("/attempt/nope")
    assert resp.status_code == 404


def test_get_attempt_200(monkeypatch):
    monkeypatch.setattr(
        persistence,
        "get_attempt_detail",
        lambda attempt_id: {"attempt_id": attempt_id, "events": [], "hints": []},
    )
    resp = client.get("/attempt/abc")
    assert resp.status_code == 200
    assert resp.json()["attempt_id"] == "abc"


# ---------- 集成：真实落库（PostgreSQL 可达时执行） ----------

@pytest.mark.real_db
@pytest.mark.skipif(not db_available(), reason="PostgreSQL 不可用（docker compose 未启动）")
def test_analyze_flow_persists_to_db(monkeypatch):
    install_fake_llm(monkeypatch)
    run = uuid.uuid4().hex[:8]
    sid = f"stu_api_{run}"

    r1 = _post(student_id=sid)
    _post(student_id=sid)  # 第二轮：触发提示升级 + 第二条事件/提示

    detail = persistence.get_attempt_detail(r1["attempt_id"])
    assert detail is not None
    assert detail["student_id"] == sid
    assert detail["problem_id"] == "phy_001"
    assert [e["event_type"] for e in detail["events"]] == ["submit", "submit"]
    assert len(detail["hints"]) == 2
    assert detail["hint_count"] == 0  # 未点过"请求提示"
    assert detail["final_correct"] is None

    # 清理本次测试数据（事件/提示随 attempt 级联删除）
    from app.db.postgres import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM attempts WHERE student_id=:s"), {"s": sid})
        conn.execute(text("DELETE FROM student_profiles WHERE student_id=:s"), {"s": sid})
        conn.execute(text("DELETE FROM students WHERE id=:s"), {"s": sid})
