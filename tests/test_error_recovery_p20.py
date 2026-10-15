"""P20 错误恢复与鲁棒性测试。

任务书的降级链：

    API timeout → 重试（P17 已有）
    模型返回空结果 / JSON 格式错误 / Structured Output 解析失败
        → 自动重试 → fallback parser → 安全默认状态
    Graph 缺失 / RAG 不可用 / 数据库暂时失败（已有）

**本文件最重要的断言不是"没崩"，而是"降级时没有编造判断"。**
一个永远不报错、但会凭空给出 `has_obvious_error=False` 的系统，
比一个诚实报错的系统更危险：学生明明卡住却拿不到帮助，而且没人知道出过错。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent.hint_policy import HintPolicyInput, decide_hint_level
from app.agent.schemas import Diagnosis, Hint, StepAnalysis
from app.db import persistence
from app.main import app
from app.services.llm.fallback import lenient_parse, safe_default
from app.services.llm.schemas import StructuredOutcome
from fakes import FakeLLM, install_fake_llm

client = TestClient(app)


class BrokenLLM(FakeLLM):
    """所有结构化调用都失败——模拟模型挂了 / 一直返回坏 JSON。"""

    def __init__(self, message: str = "模型服务不可用", **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def chat_structured(self, system: str, user: str, schema: type):
        raise RuntimeError(self.message)


def install_broken_llm(monkeypatch, message: str = "模型服务不可用") -> BrokenLLM:
    from app.agent import graph as graph_module
    from app.agent import step_matcher as step_matcher_module

    fake = BrokenLLM(message=message)
    monkeypatch.setattr(graph_module, "get_llm", lambda *a, **kw: fake)
    monkeypatch.setattr(step_matcher_module, "get_llm", lambda *a, **kw: fake)
    return fake


# ================================================================ fallback parser

@pytest.mark.parametrize(
    "label,text",
    [
        ("尾逗号", '{"current_step": "列方程", "has_obvious_error": false,}'),
        ("代码块", '```json\n{"current_step": "审题", "has_obvious_error": true}\n```'),
        ("前后有解释", '好的，结果如下：{"current_step": "审题", "has_obvious_error": false} 以上'),
        ("单引号+Python布尔", "{'current_step': '审题', 'has_obvious_error': True}"),
        ("全角标点", '{"current_step"："审题"，"has_obvious_error"：false}'),
    ],
)
def test_lenient_parse_salvages_common_malformed_json(label, text):
    parsed, _ = lenient_parse(text, StepAnalysis)

    assert parsed is not None, f"{label} 应当能抢救出来"
    assert parsed.current_step in ("列方程", "审题")


def test_lenient_parse_drops_only_the_bad_field():
    """大部分字段是对的，只有个别脏 —— 不该整体丢弃。"""
    text = '{"current_step": "列方程", "concepts": "本该是数组", "has_obvious_error": true}'

    parsed, notes = lenient_parse(text, StepAnalysis)

    assert parsed is not None
    assert parsed.current_step == "列方程"
    assert parsed.has_obvious_error is True
    assert parsed.concepts == []                       # 脏字段回落默认值
    assert any("concepts" in note for note in notes)   # 且说明了丢了什么


def test_lenient_parse_salvages_from_truncated_json():
    """被截断的输出（max_tokens 用尽）也要尽量捞出真实字段。"""
    text = '{"current_step": "列方程", "has_obvious_error": true, "concepts": ["动量守恒"'

    parsed, notes = lenient_parse(text, StepAnalysis)

    assert parsed is not None
    assert parsed.current_step == "列方程"
    assert parsed.has_obvious_error is True
    assert any("正则抢救" in note for note in notes)


def test_lenient_parse_gives_up_on_pure_prose():
    parsed, notes = lenient_parse("学生这一步在做受力分析，看起来没什么大问题", StepAnalysis)

    assert parsed is None
    assert notes


def test_lenient_parse_reports_empty_output():
    parsed, notes = lenient_parse("   ", StepAnalysis)

    assert parsed is None
    assert any("空内容" in note for note in notes)


def test_safe_default_fills_required_fields():
    """安全默认实例必须构造得出来——包括 `has_obvious_error` 这种必填字段。"""
    default = safe_default(StepAnalysis)
    assert default.current_step == ""
    assert default.has_obvious_error is False     # 类型零值
    assert safe_default(Hint).hint == ""
    assert safe_default(Diagnosis).issue_type == "none"


# ================================================================ 降级链

def test_resilient_call_never_raises(monkeypatch):
    llm = BrokenLLM("连不上")

    outcome = llm.chat_structured_resilient("sys", "user", StepAnalysis)

    assert isinstance(outcome, StructuredOutcome)
    assert outcome.degraded is True
    assert outcome.value is not None              # 拿到了安全默认实例
    assert any("连不上" in r for r in outcome.reasons)


def test_resilient_call_passes_through_success():
    outcome = FakeLLM().chat_structured_resilient("sys", "user", StepAnalysis)

    assert outcome.degraded is False
    assert outcome.value is not None
    assert outcome.reasons == []


# ================================================================ 降级时不编造

def test_degraded_round_does_not_hint():
    """降级时必须不提示：既不该凭空指责，也不该假装学生做对了。"""
    decision = decide_hint_level(
        HintPolicyInput(degraded=True, correct=False, consecutive_failures=3, request_hint=True)
    )

    assert decision.level == 0
    assert any("降级" in reason for reason in decision.reasons)


def test_graphless_degradation_does_not_fabricate_analysis(monkeypatch):
    """graphless 下 analyze_step 失败 → 不得写入任何判断字段。"""
    install_broken_llm(monkeypatch, "analyze 挂了")

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p20_a", "problem_id": "phy_001", "answer": "用机械能守恒"},
    ).json()

    assert body["degraded"] is True
    assert body["degraded_reasons"]
    assert body["hint_level"] == 0, "分析不可用时不该给提示"
    assert body["hint"] == ""
    assert body["issue_type"] in ("", "none")
    # 关键：没有把"不知道"写成"做对了"或"做错了"
    assert body["graphless"]["correct"] is None


def test_graph_mode_degradation_does_not_touch_student_state(monkeypatch):
    """匹配失败时**不能**推进 visited_nodes，也不能累加连续失败。

    空匹配的语义是"学生什么都没做对"，采用它会开始冤枉一个可能正确的学生。
    """
    install_fake_llm(monkeypatch)
    payload = {"student_id": "stu_p20_b", "problem_id": "phy_002", "answer": "弹簧恢复原长时分离"}

    good = client.post("/attempt/analyze", json=payload).json()
    assert good["visited_nodes"], "正常一轮应当推进状态"

    install_broken_llm(monkeypatch, "matcher 挂了")
    degraded = client.post("/attempt/analyze", json=payload).json()

    assert degraded["degraded"] is True
    assert degraded["visited_nodes"] == good["visited_nodes"], "降级轮不该改动已访问节点"
    assert degraded["current_node"] == good["current_node"]
    assert degraded["hint_level"] == 0


def test_degraded_round_does_not_escalate_consecutive_failures(monkeypatch):
    """连续失败计数不能被降级轮推高——否则几次模型抖动之后学生会突然收到 L4 讲解。"""
    install_broken_llm(monkeypatch, "一直挂")
    payload = {"student_id": "stu_p20_c", "problem_id": "phy_001", "answer": "随便答"}

    levels = []
    for _ in range(3):
        body = client.post("/attempt/analyze", json=payload).json()
        assert body["degraded"] is True
        levels.append(body["hint_level"])

    assert levels == [0, 0, 0], "降级轮不该把提示级别顶上去"


def test_degradation_does_not_overwrite_missing_reasoning(monkeypatch):
    """已有状态不会被降级轮清空。"""
    install_fake_llm(
        monkeypatch,
        analysis=StepAnalysis(
            current_step="给结论", has_obvious_error=False, is_complete=False,
            missing_reasoning=["未说明依据"],
        ),
    )
    payload = {"student_id": "stu_p20_d", "problem_id": "phy_001", "answer": "第一次"}
    first = client.post("/attempt/analyze", json=payload).json()
    assert first["graphless"]["missing_reasoning"] == ["未说明依据"]

    install_broken_llm(monkeypatch, "挂了")
    second = client.post("/attempt/analyze", json=payload).json()

    assert second["degraded"] is True
    assert second["graphless"]["missing_reasoning"] == ["未说明依据"], "降级不该抹掉上一轮的结论"


# ================================================================ 可观测性

@pytest.fixture
def captured_traces(monkeypatch):
    calls: list[dict] = []

    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(persistence, "persist_trace", fake)
    return calls


def test_degraded_reasons_reach_the_trace(monkeypatch, captured_traces):
    """降级原因必须进 Trace：这是区分"模型答错了"与"模型根本没答上来"的唯一线索。"""
    install_broken_llm(monkeypatch, "网关 502")

    client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p20_e", "problem_id": "phy_001", "answer": "作答"},
    )

    payloads = [call["kwargs"]["payload"] for call in captured_traces]
    assert payloads
    reasons = payloads[-1].get("degraded_reasons")
    assert reasons and any("502" in r for r in reasons)


def test_graph_exception_still_writes_a_trace(monkeypatch, captured_traces):
    """端点级兜底：图内部抛异常时，这一轮也要留下可排障的记录。

    否则最需要排障的那一轮恰恰什么都没留下，只能靠猜。
    """
    from app.api import attempts as attempts_module

    class Exploding:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("图内部炸了")

    monkeypatch.setattr(attempts_module, "tutor_graph", Exploding())

    resp = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p20_f", "problem_id": "phy_001", "answer": "作答"},
    )

    assert resp.status_code == 503
    assert "traces" in resp.json()["detail"]
    assert captured_traces, "失败轮也必须落 trace"
    payload = captured_traces[-1]["kwargs"]["payload"]
    assert payload["error"]["type"] == "RuntimeError"
    assert "图内部炸了" in payload["error"]["message"]


def test_normal_round_is_not_marked_degraded(monkeypatch):
    """别把正常轮也标成降级——那会让这个标志失去意义。"""
    install_fake_llm(monkeypatch)

    body = client.post(
        "/attempt/analyze",
        json={"student_id": "stu_p20_g", "problem_id": "phy_002", "answer": "弹簧恢复原长时分离"},
    ).json()

    assert body["degraded"] is False
    assert body["degraded_reasons"] == []
