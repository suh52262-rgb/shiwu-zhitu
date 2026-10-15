"""Trace 与可观测性测试（P19，任务书第 P19 节）。

任务书要求每次 Agent Run 都产生完整 Trace，出问题时能回答：
Step Matcher 是否先判断错了？Diagnosis 是否正确？RAG 是否检索到错误内容？
Hint Policy 是否给了错误等级？还是 Tutor 自己生成失败？

这里逐项验证这些字段确实被记录，并验证 request_id 能贯穿一次请求。
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.db import persistence
from app.main import app
from app.observability import REQUEST_ID_HEADER
from app.services.llm import OpenAICompatibleProvider, TokenUsage
from app.services.llm.schemas import LLMCall
from app.services.llm.tracking import UsageCollector, track_usage
from fakes import install_fake_llm

client = TestClient(app)

PAYLOAD = {"student_id": "001", "problem_id": "phy_001", "answer": "mgh = 1/2mv^2"}
GRAPH_PAYLOAD = {**PAYLOAD, "problem_id": "phy_002", "answer": "弹簧恢复原长时二者分离"}


# ---------------------------------------------------------------- 假 SDK（走真实 provider）

class _SchemaRoutingCompletions:
    """按 system 提示词判断当前节点，返回对应的合法 JSON。

    与 `fakes.FakeLLM` 的区别：FakeLLM 整个替换掉 `get_llm`，绕过了 provider 层；
    这里让**真实的 OpenAICompatibleProvider** 跑起来（只替换底层 SDK），
    因此 token 用量、模型名、延迟归集这些 P19 关心的机制都会被真正执行。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if "步骤匹配器" in system:
            payload = {
                "matched_nodes": ["S5"],
                "attempted_nodes": [],
                "confidence": 0.9,
                "correct": True,
                "current_method": "能量法",
                "missing_reasoning": [],
                "off_graph": False,
                "explanation": "假 SDK",
            }
        elif "分析学生的解题进展" in system:
            payload = {
                "current_step": "建立方程",
                "method": "能量法",
                "has_obvious_error": False,
                "concepts": ["机械能守恒"],
                "is_complete": False,
            }
        elif "归因" in system:
            payload = {
                "issue_type": "none",
                "issue": "",
                "concept": "",
                "severity": "low",
                "recommended_action": "",
            }
        else:
            payload = {"hint": "再检查一下分离条件。"}

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15),
        )


class _FakeSDK:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SchemaRoutingCompletions())


@pytest.fixture
def real_provider_llm(monkeypatch):
    """装上真实 provider（底层假 SDK），用于验证归集与落库链路。"""
    from app.agent import graph as graph_module
    from app.agent import step_matcher as step_matcher_module

    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://fake.local",
        model="fake-model",
        name="deepseek",
        client=_FakeSDK(),
    )
    monkeypatch.setattr(graph_module, "get_llm", lambda *a, **k: provider)
    monkeypatch.setattr(step_matcher_module, "get_llm", lambda *a, **k: provider)
    return provider


@pytest.fixture
def captured_traces(monkeypatch):
    """拦下 persist_trace，返回 (位置参数, 关键字参数) 列表。"""
    calls: list[dict] = []

    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(persistence, "persist_trace", fake)
    return calls


def _post(**overrides) -> dict:
    return client.post("/attempt/analyze", json={**PAYLOAD, **overrides}).json()


# ---------------------------------------------------------------- request id

def test_request_id_is_generated_and_echoed():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get(REQUEST_ID_HEADER)
    assert len(resp.headers[REQUEST_ID_HEADER]) == 16


def test_request_id_from_client_is_respected():
    resp = client.get("/health", headers={REQUEST_ID_HEADER: "trace-me-123"})
    assert resp.headers[REQUEST_ID_HEADER] == "trace-me-123"


def test_request_id_is_unique_per_request():
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]
    assert first != second


def test_request_id_reaches_the_endpoint_and_the_trace(monkeypatch, captured_traces):
    """中间件写入的 request_id 必须能在同步端点（线程池）里读到并落进 trace。"""
    install_fake_llm(monkeypatch)

    client.post(
        "/attempt/analyze",
        json=PAYLOAD,
        headers={REQUEST_ID_HEADER: "round-abc-999"},
    )

    assert captured_traces, "没有写入 trace"
    assert captured_traces[0]["kwargs"]["request_id"] == "round-abc-999"


# ---------------------------------------------------------------- trace 字段

def test_trace_records_task_required_fields(monkeypatch, captured_traces):
    """任务书 P19 列出的字段必须都在。"""
    install_fake_llm(monkeypatch)
    _post()

    assert captured_traces
    trace = captured_traces[0]
    payload = trace["kwargs"]["payload"]

    # 独立列
    assert trace["kwargs"]["problem_id"] == "phy_001"
    assert trace["kwargs"]["request_id"]
    assert trace["kwargs"]["latency_ms"] >= 0
    assert "token_usage" in trace["kwargs"]
    assert "prompt_versions" in trace["kwargs"]
    assert trace["kwargs"]["before_state"]

    # payload 内的链路字段
    for key in (
        "student_input",
        "learning_events",
        "step_match",
        "diagnosis",
        "stuck_score",
        "rag_query",
        "rag_context",
        "hint_policy",
        "tutor_output",
        "after_state",
    ):
        assert key in payload, f"trace 缺少 {key}"


def test_trace_before_and_after_state_show_the_transition(monkeypatch, captured_traces):
    """用有 Graph 的题：before 无已完成节点，after 记下本轮到达的节点。"""
    install_fake_llm(monkeypatch)
    client.post("/attempt/analyze", json=GRAPH_PAYLOAD)

    trace = captured_traces[0]["kwargs"]
    before, after = trace["before_state"], trace["payload"]["after_state"]

    assert before["visited_nodes"] == []          # 第一轮前没有任何已完成节点
    assert before["hint_level"] == 0
    assert after["visited_nodes"] == ["S5"]
    assert after["current_node"] == "S5"
    assert after["hint_level"] == trace["payload"]["hint_policy"]["level"]


def test_trace_records_learning_events(monkeypatch, captured_traces):
    install_fake_llm(monkeypatch)
    _post(request_hint=True)

    events = captured_traces[0]["kwargs"]["payload"]["learning_events"]
    assert [e["event_type"] for e in events] == ["submit", "hint_request"]


def test_trace_records_hint_policy_decision(monkeypatch, captured_traces):
    install_fake_llm(monkeypatch)
    body = _post()
    policy = captured_traces[0]["kwargs"]["payload"]["hint_policy"]

    assert policy["level"] == body["hint_level"]
    assert isinstance(policy["reasons"], list) and policy["reasons"]
    assert "consecutive_failures" in policy
    assert "same_hint_before" in policy


def test_hint_policy_correct_reflects_this_round_not_the_previous_one(monkeypatch, captured_traces):
    """回归：correct/off_graph/consecutive_failures 必须取**本轮结束后**的状态。

    曾经误用本轮开始前的 state，导致 trace 里 correct 整整错开一轮——
    表现为 reasons 写着"本轮推进正确"而 correct 却是 False。
    """
    install_fake_llm(monkeypatch)          # 默认假 LLM：正确匹配 S5
    client.post("/attempt/analyze", json=GRAPH_PAYLOAD)

    trace = captured_traces[0]["kwargs"]
    policy = trace["payload"]["hint_policy"]

    assert trace["before_state"]["visited_nodes"] == []     # 本轮前确实没有推进过
    assert policy["correct"] is True                        # 本轮推进正确
    assert policy["consecutive_failures"] == 0
    assert policy["reasons"], "策略必须给出可解释理由"
    # 默认假 LLM（STEP_MATCH_S5）带 missing_reasoning，所以本轮是
    # "结论正确但缺关键推理" → L1 要求补齐推理链（审计修复后的行为）
    assert policy["level"] >= 1
    assert any("推理" in reason for reason in policy["reasons"])


def test_trace_records_model_and_token_usage(real_provider_llm, captured_traces):
    """P19：模型名与 token 用量来自 P17 的调用归集器（走真实 provider）。"""
    _post()

    trace = captured_traces[0]["kwargs"]

    assert trace["model"] == "fake-model"
    assert trace["provider"] == "deepseek"
    assert trace["token_usage"]["calls"] >= 1
    assert trace["token_usage"]["total_tokens"] == 15 * trace["token_usage"]["calls"]
    assert trace["token_usage"]["models"] == ["fake-model"]
    assert trace["llm_latency_ms"] >= 0


def test_trace_usage_is_empty_when_llm_layer_is_bypassed(monkeypatch, captured_traces):
    """FakeLLM 直接替换 get_llm，绕过 provider 层——归集器为空是预期行为。

    这条用例把该边界固定下来，免得以后误以为"字段没值就是坏了"。
    """
    install_fake_llm(monkeypatch)
    _post()

    trace = captured_traces[0]["kwargs"]
    # **显式关闭检索重排**：它默认开启并且会调模型
    # （实测 +5.3%、p=0.0009，是唯一有效的检索改动）。
    # 这条测试的前提是"检索路径不碰模型"，所以要先把重排关掉——
    # 而不是放宽断言，那样就测不出"绕过 LLM 层时用量为空"这件事了。
    from app.rag import rerank as _rerank
    import app.agent.tools as _tools  # noqa: F401  确保模块已加载

    monkeypatch.setattr(_rerank, "ENABLED", False)
    assert trace["token_usage"]["calls"] == 0
    assert trace["model"] == ""


def test_round_index_counts_submissions_not_hint_requests(monkeypatch, captured_traces):
    """轮次只数 submit：点几次"请求提示"不该让轮号跳号。"""
    install_fake_llm(monkeypatch)
    _post()
    _post(request_hint=True)
    _post(request_hint=True)

    rounds = [c["kwargs"]["round_index"] for c in captured_traces]
    assert rounds == [1, 2, 3]


# ---------------------------------------------------------------- 归集器

def test_usage_collector_aggregates_calls():
    collector = UsageCollector()
    collector.add(
        LLMCall(provider="deepseek", model="cheap", latency_ms=100,
                usage=TokenUsage(10, 5, 15))
    )
    collector.add(
        LLMCall(provider="deepseek", model="strong", latency_ms=200, attempts=2,
                usage=TokenUsage(20, 8, 28))
    )

    data = collector.as_dict()
    assert data["calls"] == 2
    assert data["total_tokens"] == 43
    assert data["llm_latency_ms"] == 300
    assert data["models"] == ["cheap", "strong"]
    assert data["retried_calls"] == 1
    assert collector.model_label() == "cheap+strong"


def test_usage_collector_dedupes_models():
    collector = UsageCollector()
    for _ in range(3):
        collector.add(LLMCall(provider="deepseek", model="same", usage=TokenUsage(1, 1, 2)))

    assert collector.models == ["same"]
    assert collector.model_label() == "same"
    assert collector.total_usage.total_tokens == 6


def test_track_usage_is_scoped_and_nestable():
    from app.services.llm.tracking import current_collector

    assert current_collector() is None
    with track_usage() as outer:
        outer.add(LLMCall(provider="p", model="m", usage=TokenUsage(1, 1, 2)))
        with track_usage() as inner:
            inner.add(LLMCall(provider="p", model="m", usage=TokenUsage(3, 3, 6)))
            assert current_collector() is inner
        assert current_collector() is outer
        assert outer.total_usage.total_tokens == 2      # 内层的调用不进外层
    assert current_collector() is None


def test_provider_calls_land_in_the_active_collector():
    """provider → 归集器 的接线（P19 的 token/model 就靠这条链路）。"""
    provider = OpenAICompatibleProvider(
        api_key="k",
        base_url="http://fake.local",
        model="wired-model",
        name="qwen",
        client=_FakeSDK(),
    )

    with track_usage() as collector:
        provider.generate_text([{"role": "user", "content": "hi"}])

    assert len(collector.calls) == 1
    assert collector.models == ["wired-model"]
    assert collector.providers == ["qwen"]
    assert collector.total_usage.total_tokens == 15


def test_collector_outside_tracking_is_noop():
    provider = OpenAICompatibleProvider(
        api_key="k", base_url="http://x", model="m", client=_FakeSDK()
    )
    provider.generate_text([{"role": "user", "content": "hi"}])   # 不应抛异常
    assert provider.last_call is not None


# ---------------------------------------------------------------- 查询接口

def test_traces_endpoint_503_when_db_down(monkeypatch):
    def unavailable(attempt_id):
        raise persistence.PersistenceUnavailable("connection refused")

    monkeypatch.setattr(persistence, "get_attempt_row", unavailable)
    assert client.get("/attempt/x/traces").status_code == 503


def test_traces_endpoint_404_when_attempt_missing(monkeypatch):
    monkeypatch.setattr(persistence, "get_attempt_row", lambda attempt_id: None)
    assert client.get("/attempt/nope/traces").status_code == 404


def test_traces_endpoint_returns_rounds(monkeypatch):
    monkeypatch.setattr(persistence, "get_attempt_row", lambda attempt_id: {"attempt_id": attempt_id})
    monkeypatch.setattr(
        persistence,
        "get_traces",
        lambda attempt_id: [
            {"round_index": 1, "request_id": "r1", "payload": {"student_input": "第一步"}},
            {"round_index": 2, "request_id": "r2", "payload": {"student_input": "第二步"}},
        ],
    )

    body = client.get("/attempt/abc/traces").json()

    assert body["count"] == 2
    assert [t["round_index"] for t in body["traces"]] == [1, 2]
    assert body["traces"][0]["request_id"] == "r1"
