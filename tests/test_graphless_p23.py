"""P23 Graphless Mode 强化测试。

任务书 P23 三部分：
1. 至少输出 {current_stage, method, correct, issue_type, knowledge_points,
   missing_reasoning, recommended_next_step}；
2. 要真的能用（判断阶段 / 是否明显错误 / 相关知识点 / 是否该提示），不是占位；
3. 预留与 Auto Graph 的联动：
   新题 → graphless 立即可用 → 后台生成候选图 → 验证通过 → **后续 attempt 自动切 graph 模式**。

第 3 条的关键性质是：切换必须**自动发生**，不需要重启、不需要改代码、
也不需要客户端做任何事。它由"每轮请求都重新解析该题有没有图"保证。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.graphless import GraphlessAssessment, build_assessment
from app.models.student import StudentState
from fakes import install_fake_llm

client = TestClient(app)


# ---------------------------------------------------------------- 1. 结构化输出

def test_build_assessment_maps_all_seven_fields():
    """任务书列出的七个字段一个都不能少。"""
    state = StudentState(
        current_step="建立方程",
        current_method="动量守恒",
        has_obvious_error=False,
        issue_type="none",
        concepts=["动量守恒", "相对速度"],
        missing_reasoning=["未说明合外力为零"],
        recommended_next_step="先交代使用动量守恒的依据",
    )

    a = build_assessment(state)

    assert a.current_stage == "建立方程"
    assert a.method == "动量守恒"
    assert a.correct is True
    assert a.issue_type is None                       # "none" 归一化成 None
    assert a.knowledge_points == ["动量守恒", "相对速度"]
    assert a.missing_reasoning == ["未说明合外力为零"]
    assert a.recommended_next_step == "先交代使用动量守恒的依据"


def test_build_assessment_correct_is_tri_state():
    """`correct=None`（无法判定）必须与 `False`（判定为错）区分开。

    把"不知道"当成"做错了"，会让系统对一个可能完全正确的学生一路升级提示。
    """
    unknown = build_assessment(StudentState(has_obvious_error=None))
    wrong = build_assessment(StudentState(has_obvious_error=True))

    assert unknown.correct is None
    assert wrong.correct is False


def test_build_assessment_accepts_plain_dict():
    """API 层传进来的是图的运行结果（dict），不是 StudentState 实例。"""
    a = build_assessment({"current_step": "审题", "has_obvious_error": False, "concepts": ["力学"]})

    assert a.current_stage == "审题"
    assert a.correct is True
    assert a.knowledge_points == ["力学"]


def test_build_assessment_normalizes_real_issue_type():
    a = build_assessment(StudentState(has_obvious_error=True, issue_type="concept"))
    assert a.issue_type == "concept"


def test_graphless_assessment_defaults_are_safe():
    a = GraphlessAssessment()
    assert a.correct is None
    assert a.issue_type is None
    assert a.knowledge_points == []
    assert a.missing_reasoning == []
    assert a.recommended_next_step == ""


# ---------------------------------------------------------------- 2. graphless 真的能用

def test_graphless_response_carries_assessment(monkeypatch):
    """无 Graph 的题目，响应里必须带 graphless 判断块。"""
    install_fake_llm(monkeypatch)

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p23_a", "problem_id": "phy_001", "answer": "用机械能守恒"},
    ).json()

    assert body["graph_mode"] is False
    assert body["graphless"] is not None
    assert set(body["graphless"]) >= {
        "current_stage", "method", "correct", "issue_type",
        "knowledge_points", "missing_reasoning", "recommended_next_step",
    }


def test_graph_mode_response_has_no_graphless_block(monkeypatch):
    """有 Graph 时不给 graphless 块——判断是节点锚定的，语义不同，不该混。"""
    install_fake_llm(monkeypatch)

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p23_b", "problem_id": "phy_002", "answer": "弹簧恢复原长时分离"},
    ).json()

    assert body["graph_mode"] is True
    assert body["graphless"] is None
    assert body["step_match"] is not None     # graph 模式的信息在 step_match 里


def test_graphless_surfaces_missing_reasoning(monkeypatch):
    """P23 核心修复：graphless 下 missing_reasoning 不再是恒空。

    此前只有 Step Matcher（graph 模式）会填这个字段，于是
    "结论对但跳过推理"的学生在无图题目上拿不到任何反馈。
    """
    from app.agent.schemas import StepAnalysis

    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="给出结论",
            method="动量守恒",
            has_obvious_error=False,
            concepts=["动量守恒"],
            is_complete=False,
            missing_reasoning=["未说明为何可对整体使用动量守恒"],
            recommended_next_step="先交代系统合外力为零，再列守恒式",
        ),
    )

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p23_c", "problem_id": "phy_001", "answer": "分离时 N=0"},
    ).json()

    g = body["graphless"]
    assert g["missing_reasoning"] == ["未说明为何可对整体使用动量守恒"]
    assert g["recommended_next_step"] == "先交代系统合外力为零，再列守恒式"
    # 结论正确但缺推理 → 必须给提示（这条链路此前在 graphless 下是断的）
    assert body["hint_level"] >= 1
    assert body["hint"]


def test_graphless_clears_missing_reasoning_when_reasoning_complete(monkeypatch):
    """学生补齐推理后，上一轮的缺失项必须清掉（字段语义是"最近一轮"）。"""
    from app.agent.schemas import StepAnalysis

    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="补齐推理", has_obvious_error=False, is_complete=False,
            missing_reasoning=["缺少依据"],
        ),
    )
    payload = {"student_id": "stu_p23_d", "problem_id": "phy_001", "answer": "第一次作答"}
    first = client.post("/attempt/analyze", json=payload).json()
    assert first["graphless"]["missing_reasoning"] == ["缺少依据"]

    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="补齐推理", has_obvious_error=False, is_complete=False,
            missing_reasoning=[],
        ),
    )
    second = client.post("/attempt/analyze", json=payload).json()
    assert second["graphless"]["missing_reasoning"] == []


def test_analyze_step_does_not_overwrite_step_matcher_missing_reasoning(monkeypatch):
    """graph 模式下，analyze_step **不能**覆盖 Step Matcher 的节点级结论。

    两者的执行顺序是 step_matching → analyze_step，若 analyze_step 也写这个字段，
    就会把更精确的结论冲掉。
    """
    from app.agent.schemas import StepAnalysis

    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="x", has_obvious_error=True, is_complete=False,
            missing_reasoning=["来自 analyze_step 的粗糙结论"],
        ),
    )

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p23_e", "problem_id": "phy_002", "answer": "弹簧原长时分离"},
    ).json()

    # 默认假 Step Matcher 给的是节点级结论
    assert body["step_match"]["missing_reasoning"] != ["来自 analyze_step 的粗糙结论"]


# ---------------------------------------------------------------- 3. 与 Auto Graph 联动

@pytest.fixture
def graphless_problem(tmp_path, monkeypatch):
    """一道只有题面、没有图的题（以及它的目录，便于中途补图）。"""
    problem_id = "phy_sw"
    target = tmp_path / problem_id
    target.mkdir(parents=True)
    (target / "problem.json").write_text(
        json.dumps(
            {"problem_id": problem_id, "title": "联动测试题", "content": "求分离速度"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROBLEMS_DIR", str(tmp_path))
    return target


def _write_graph(target, problem_id):
    graph = {
        "problem_id": problem_id,
        "nodes": [
            {"id": "S1", "name": "分析受力", "type": "modeling", "description": "受力分析"},
            {"id": "S2", "name": "求分离速度", "type": "result", "description": "由 N=0 求速度"},
        ],
        "edges": [["S1", "S2"]],
        "valid_paths": [["S1", "S2"]],
    }
    (target / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")


def test_attempt_switches_to_graph_mode_when_graph_appears(graphless_problem, monkeypatch):
    """P23 联动：新题先用 graphless 跑，图补上后**同一个 attempt** 自动切到 graph 模式。

    这条性质不需要任何额外机制——API 每轮都重新解析"这道题有没有图"，
    所以图一出现就自动生效。测试把它固定下来，防止将来有人加缓存把它破坏掉。
    """
    install_fake_llm(monkeypatch)
    payload = {"student_id": "stu_sw", "problem_id": "phy_sw", "answer": "先分析受力"}

    first = client.post("/attempt/analyze", json=payload).json()
    assert first["graph_mode"] is False
    assert first["graphless"] is not None
    assert first["step_match"] is None

    # 模拟"后台生成候选图并通过校验"
    _write_graph(graphless_problem, "phy_sw")

    second = client.post("/attempt/analyze", json=payload).json()
    assert second["attempt_id"] == first["attempt_id"], "切换不该另起一个 attempt"
    assert second["graph_mode"] is True
    assert second["graphless"] is None
    assert second["step_match"] is not None


@pytest.fixture
def captured_traces(monkeypatch):
    """拦下 persist_trace。

    `/attempt/{id}/traces` 读的是 PostgreSQL，而测试用的是内存 AttemptStore
    （conftest 的 isolated_attempt_store），所以没法回头查库——
    直接拦下落库调用，看写进去的 payload 更贴近被测代码。
    """
    from app.db import persistence

    calls: list[dict] = []

    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(persistence, "persist_trace", fake)
    return calls


def test_trace_records_round_mode_so_the_switch_is_visible(
    graphless_problem, monkeypatch, captured_traces
):
    """切换点必须能在 Trace 里看出来——否则排障时分不清哪轮用了哪套逻辑。"""
    install_fake_llm(monkeypatch)
    payload = {"student_id": "stu_sw2", "problem_id": "phy_sw", "answer": "先分析受力"}

    client.post("/attempt/analyze", json=payload)
    _write_graph(graphless_problem, "phy_sw")
    client.post("/attempt/analyze", json=payload)

    modes = [call["kwargs"]["payload"].get("graph_mode") for call in captured_traces]

    assert modes == [False, True]
    # 轮次编号也要连续，否则"第几轮切换的"说不清
    assert [call["kwargs"]["round_index"] for call in captured_traces] == [1, 2]
