"""Graphless Mode 测试（P8，任务书第 11 节）。

Graphless = 题目没有 Problem Graph 时仍可运行基本功能：
LLM 动态判断当前步骤 / 方法 / 错误 / 卡顿知识点，并给出提示。
"""
from fastapi.testclient import TestClient

from app.agent.schemas import StepAnalysis
from app.main import app
from app.services.problem_store import has_problem_graph
from fakes import BAD_ANALYSIS, GOOD_ANALYSIS, install_fake_llm

client = TestClient(app)


def test_has_problem_graph_branching():
    assert has_problem_graph("phy_002") is True
    assert has_problem_graph("phy_001") is False
    assert has_problem_graph("no_such") is False


def test_graph_endpoint_returns_graph_for_pq_problem():
    resp = client.get("/problems/phy_002/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert body["problem_id"] == "phy_002"
    assert len(body["nodes"]) == 12
    assert {"S1", "S11"} <= {n["id"] for n in body["nodes"]}


def test_graph_endpoint_404_for_graphless_problem():
    resp = client.get("/problems/phy_001/graph")
    assert resp.status_code == 404
    assert "graphless" in resp.json()["detail"]


def test_graphless_multi_round_session_still_works(monkeypatch):
    """无 Graph 的题（phy_001）连续多轮：分析、归因、提示、完成全链可用。"""
    install_fake_llm(monkeypatch)
    payload = {"student_id": "stu_p8", "problem_id": "phy_001"}

    r1 = client.post("/attempt/analyze", json={**payload, "answer": "mgh = 1/2mv^2"}).json()
    assert r1["graph_mode"] is False
    assert r1["issue_type"] == "concept"
    assert r1["hint_level"] == 1

    install_fake_llm(monkeypatch, analysis=GOOD_ANALYSIS)
    r2 = client.post(
        "/attempt/analyze", json={**payload, "answer": "W_f = 1/2mv^2 - mgh，不守恒"}
    ).json()
    assert r2["finished"] is True  # graphless 完成判定：is_complete 且无错误
    assert r2["current_node"] == ""  # 无 Graph，全程没有节点概念


def test_graphless_hint_escalation_unaffected(monkeypatch):
    """graphless 模式下 Hint Policy 的连续失败升级照常工作。"""
    install_fake_llm(monkeypatch)
    payload = {"student_id": "stu_p8b", "problem_id": "phy_001", "answer": "错答"}
    levels = [
        client.post("/attempt/analyze", json=payload).json()["hint_level"]
        for _ in range(4)
    ]
    assert levels == [1, 2, 3, 4]


def test_graphless_correct_answer_is_not_treated_as_failure(monkeypatch):
    """回归：graphless 下"答对"必须判为推进正确——级别归零、失败计数不累积。

    历史缺陷：`graphless_mode_matching` 用 `state.has_obvious_error is False` 判断对错，
    而该字段是**上一轮**的分析结论、且 API 从未把它传进图内（恒为 None），
    于是 `None is False` **恒为 False**：学生答对也被当成连续失败，
    提示级别 L1→L2→L3 只升不降，第 4 轮起"主动请求即 L5 完整讲解"。
    实测过端点：连续 3 轮答对时 hint_level 曾是 1/2/3，而 issue_type 一直是 'none'。
    """
    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="继续求解",
            method="动能定理",
            has_obvious_error=False,
            concepts=[],
            is_complete=False,      # 未完成，保持同一个 attempt 连续三轮
        ),
    )
    payload = {"student_id": "stu_gl_ok", "problem_id": "phy_001", "answer": "正确的一步"}

    levels = []
    for _ in range(3):
        body = client.post("/attempt/analyze", json=payload).json()
        assert body["issue_type"] == "none"      # 没有诊断出错误
        levels.append(body["hint_level"])

    assert levels == [0, 0, 0], "答对不应触发任何提示"
