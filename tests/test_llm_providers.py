"""LLM provider 抽象层测试（P17）。

全部使用假 SDK，不发起网络请求。覆盖：
- 统一的文本 / 结构化输出入口（消息列表形式）
- 错误分类（超时 / 限流 / 网络 / 5xx / 4xx）
- 重试策略（该重试的重试，不该重试的立刻抛）
- token 用量、模型名、provider 元信息（P19 Trace 依赖这些）
- Anthropic 适配（system 顶层参数、content 块拼接、usage 字段名差异）
- 工厂：兼容网关、按节点换模型
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import get_settings
from app.services.llm import (
    AnthropicProvider,
    LLMConfigError,
    LLMNetworkError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    OpenAICompatibleProvider,
    TokenUsage,
    classify_exception,
    get_llm,
    is_retryable,
    messages,
    usage_from_response,
)
from app.services.llm import factory as factory_module
from app.services.llm.base import _append_instruction, _append_to_last_user


class StepAnalysis(BaseModel):
    current_step: str
    has_obvious_error: bool


# ---------------------------------------------------------------- 假 SDK

class FakeAPIError(Exception):
    """模拟 openai / anthropic 的 APIStatusError（带 status_code）。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeRateLimitError(FakeAPIError):
    """类名含 RateLimit，模拟 openai.RateLimitError。"""


class FakeAuthenticationError(FakeAPIError):
    """401：不该重试。"""


class FakeInternalServerError(FakeAPIError):
    """500：该重试。"""


class FakeAPITimeoutError(Exception):
    """类名含 Timeout，模拟 openai.APITimeoutError。"""


class FakeAPIConnectionError(Exception):
    """类名含 Connection，模拟 openai.APIConnectionError。"""


def _usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


class FakeCompletions:
    def __init__(self, responses) -> None:
        # 元素可以是 str，也可以是 Exception（模拟调用失败）
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
            usage=_usage(),
        )


class FakeOpenAI:
    def __init__(self, responses) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class FakeAnthropicMessages:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=item)],
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


class FakeAnthropic:
    def __init__(self, responses) -> None:
        self.messages = FakeAnthropicMessages(responses)


def openai_provider(responses, **kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="k",
        base_url="http://fake.local",
        model="test-model",
        name="deepseek",
        client=FakeOpenAI(responses),
        **kwargs,
    )


def anthropic_provider(responses, **kwargs) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="k", model="claude-test", client=FakeAnthropic(responses), **kwargs
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """重试退避不该拖慢测试。"""
    monkeypatch.setattr("app.services.llm.base.time.sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _reset_caches():
    get_settings.cache_clear()
    factory_module.clear_cache()
    yield
    get_settings.cache_clear()
    factory_module.clear_cache()


# ---------------------------------------------------------------- 文本生成

def test_generate_text_accepts_message_list():
    llm = openai_provider(["机械能守恒的条件是只有重力做功。"])
    out = llm.generate_text(messages("你是助教", "判断是否守恒"))

    assert "机械能守恒" in out
    sent = llm._client.chat.completions.calls[0]["messages"]
    assert sent == [
        {"role": "system", "content": "你是助教"},
        {"role": "user", "content": "判断是否守恒"},
    ]


def test_generate_text_without_system_message():
    llm = openai_provider(["ok"])
    llm.generate_text(messages(user="只有用户消息"))
    assert llm._client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "只有用户消息"}
    ]


def test_legacy_chat_still_works():
    llm = openai_provider(["答复"])
    assert llm.chat("sys", "usr") == "答复"


# ---------------------------------------------------------------- 结构化输出

def test_generate_structured_parses_and_injects_schema_instruction():
    payload = {"current_step": "建立方程", "has_obvious_error": True}
    llm = openai_provider([json.dumps(payload, ensure_ascii=False)])

    result = llm.generate_structured(messages("分析作答", "mgh = 1/2mv^2"), StepAnalysis)

    assert result.current_step == "建立方程"
    system = llm._client.chat.completions.calls[0]["messages"][0]["content"]
    assert "JSON Schema" in system
    assert "has_obvious_error" in system          # schema 内容进了提示词


def test_generate_structured_retries_with_feedback_on_bad_json():
    payload = {"current_step": "受力分析", "has_obvious_error": False}
    llm = openai_provider(["这不是 JSON", json.dumps(payload, ensure_ascii=False)])

    result = llm.generate_structured(messages("s", "u"), StepAnalysis)

    assert result.current_step == "受力分析"
    calls = llm._client.chat.completions.calls
    assert len(calls) == 2
    assert "无法解析" in calls[1]["messages"][-1]["content"]   # 反馈追加到最后一条 user


def test_generate_structured_raises_after_two_bad_outputs():
    llm = openai_provider(["坏", "还是坏"])
    with pytest.raises(LLMStructuredOutputError):
        llm.generate_structured(messages("s", "u"), StepAnalysis)


def test_structured_retry_is_independent_from_network_retry():
    """网络重试 + 格式重试：共 3 次请求，且错在第 3 次才成功。"""
    payload = {"current_step": "ok", "has_obvious_error": False}
    llm = openai_provider(
        [FakeAPITimeoutError("boom"), "坏 JSON", json.dumps(payload)], max_retries=1
    )

    result = llm.generate_structured(messages("s", "u"), StepAnalysis)

    assert result.current_step == "ok"
    assert len(llm._client.chat.completions.calls) == 3


# ---------------------------------------------------------------- 错误分类

@pytest.mark.parametrize(
    "exc,expected",
    [
        (FakeAPITimeoutError("slow"), LLMTimeoutError),
        (FakeRateLimitError("too many", status_code=429), LLMRateLimitError),
        (FakeAPIConnectionError("refused"), LLMNetworkError),
        (FakeInternalServerError("boom", status_code=500), LLMResponseError),
        (ValueError("weird"), LLMResponseError),
    ],
)
def test_classify_exception(exc, expected):
    assert isinstance(classify_exception(exc), expected)


def test_classify_passes_through_domain_errors():
    original = LLMConfigError("已经是领域异常")
    assert classify_exception(original) is original


def test_is_retryable_matrix():
    assert is_retryable(LLMTimeoutError("t")) is True
    assert is_retryable(LLMRateLimitError("r")) is True
    assert is_retryable(LLMNetworkError("n")) is True
    assert is_retryable(LLMResponseError("5xx", status_code=503)) is True
    assert is_retryable(LLMResponseError("4xx", status_code=400)) is False
    assert is_retryable(LLMConfigError("配置")) is False


# ---------------------------------------------------------------- 重试行为

def test_retries_on_retryable_error_then_succeeds():
    llm = openai_provider([FakeAPITimeoutError("第一次超时"), "成功"], max_retries=2)

    assert llm.generate_text(messages("s", "u")) == "成功"
    assert len(llm._client.chat.completions.calls) == 2


def test_retries_on_server_error():
    llm = openai_provider([FakeInternalServerError("500", status_code=500), "成功"], max_retries=1)
    assert llm.generate_text(messages("s", "u")) == "成功"
    assert len(llm._client.chat.completions.calls) == 2


def test_gives_up_after_max_retries():
    llm = openai_provider(
        [FakeAPITimeoutError("1"), FakeAPITimeoutError("2")], max_retries=1
    )

    with pytest.raises(LLMTimeoutError):
        llm.generate_text(messages("s", "u"))
    assert len(llm._client.chat.completions.calls) == 2      # 首次 + 1 次重试


def test_does_not_retry_auth_error():
    """401 重试只是浪费时间，必须立刻抛。"""
    llm = openai_provider(
        [FakeAuthenticationError("bad key", status_code=401), "不该被用到"], max_retries=3
    )

    with pytest.raises(LLMResponseError):
        llm.generate_text(messages("s", "u"))
    assert len(llm._client.chat.completions.calls) == 1


def test_max_retries_zero_means_single_attempt():
    llm = openai_provider([FakeAPITimeoutError("x")], max_retries=0)
    with pytest.raises(LLMTimeoutError):
        llm.generate_text(messages("s", "u"))
    assert len(llm._client.chat.completions.calls) == 1


# ---------------------------------------------------------------- 元信息

def test_last_call_records_model_provider_usage_and_attempts():
    llm = openai_provider([FakeAPITimeoutError("重试一次"), "成功"], max_retries=1)

    llm.generate_text(messages("s", "u"))
    call = llm.last_call

    assert call is not None
    assert call.provider == "deepseek"
    assert call.model == "test-model"
    assert call.attempts == 2
    assert call.usage.total_tokens == 15          # 假 SDK 固定 10 + 5
    assert call.latency_ms >= 0
    assert call.as_dict()["token_usage"]["prompt_tokens"] == 10


def test_last_call_is_per_thread():
    """P19 依赖"本线程最近一次调用"，避免并发请求互相串数据。"""
    import threading

    llm = openai_provider(["主线程", "子线程"])

    llm.generate_text(messages(user="a"))
    assert llm.last_call is not None

    seen = {}

    def worker():
        seen["before"] = llm.last_call
        llm.generate_text(messages(user="b"))
        seen["after"] = llm.last_call

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["before"] is None                 # 子线程看不到主线程的记录
    assert seen["after"] is not None


def test_reset_last_call():
    llm = openai_provider(["x"])
    llm.generate_text(messages(user="a"))
    assert llm.last_call is not None

    llm.reset_last_call()
    assert llm.last_call is None


def test_usage_totals_accumulate():
    llm = openai_provider(["a", "b"])
    llm.generate_text(messages(user="1"))
    llm.generate_text(messages(user="2"))

    assert llm.usage_totals().total_tokens == 30


def test_describe_returns_provider_info():
    llm = openai_provider(["x"], timeout=12.5, max_retries=1)
    info = llm.describe()

    assert info.name == "deepseek"
    assert info.kind == "openai_compatible"
    assert info.model == "test-model"
    assert info.timeout == 12.5
    assert info.as_dict()["max_retries"] == 1


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider(model="m")  # type: ignore[abstract]


# ---------------------------------------------------------------- usage 解析

def test_usage_from_response_handles_openai_shape():
    usage = usage_from_response(_usage(100, 20))
    assert usage == TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)


def test_usage_from_response_handles_anthropic_shape():
    usage = usage_from_response(SimpleNamespace(input_tokens=30, output_tokens=9))
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 9
    assert usage.total_tokens == 39


def test_usage_from_response_tolerates_missing_usage():
    """测试假件与部分网关不返回 usage——绝不能因此让调用失败。"""
    assert usage_from_response(None).is_empty
    assert usage_from_response(SimpleNamespace()).is_empty


def test_usage_from_response_accepts_dict():
    usage = usage_from_response({"prompt_tokens": 3, "completion_tokens": 4})
    assert usage.total_tokens == 7


def test_usage_details_are_read_from_their_own_containers():
    """回归：cached 在 prompt_tokens_details、reasoning 在 completion_tokens_details。

    曾经用 `A or B` 只取其中一个对象、再从它身上读两个字段，
    导致两者同时存在时 reasoning_tokens 恒为 0（它按输出计费，会低估成本）。
    """
    usage = usage_from_response(
        SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=300),
        )
    )

    assert usage.cached_tokens == 800
    assert usage.reasoning_tokens == 300


def test_usage_details_tolerate_missing_containers():
    usage = usage_from_response(
        SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
        )
    )
    assert usage.cached_tokens == 0
    assert usage.reasoning_tokens == 7


def test_usage_details_accept_plain_dicts():
    usage = usage_from_response(
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    )
    assert usage.cached_tokens == 4
    assert usage.reasoning_tokens == 2


def test_usage_add_is_immutable():
    total = TokenUsage(1, 2, 3) + TokenUsage(10, 20, 30)
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (11, 22, 33)


# ---------------------------------------------------------------- 未接入能力

def test_stream_and_tools_are_declared_but_not_implemented():
    llm = openai_provider(["x"])
    with pytest.raises(NotImplementedError, match="流式输出"):
        llm.generate_stream(messages(user="a"))
    with pytest.raises(NotImplementedError, match="Tool Calling"):
        llm.call_tools(messages(user="a"), tools=[])


# ---------------------------------------------------------------- Anthropic 适配

def test_anthropic_extracts_system_to_top_level():
    llm = anthropic_provider(["答复"])
    llm.generate_text(messages("你是助教", "你好"))

    call = llm._client.messages.calls[0]
    assert call["system"] == "你是助教"                     # system 不在 messages 里
    assert call["messages"] == [{"role": "user", "content": "你好"}]
    assert call["max_tokens"] > 0                          # Anthropic 必填


def test_anthropic_works_without_system_message():
    llm = anthropic_provider(["答复"])
    llm.generate_text(messages(user="只有用户"))

    call = llm._client.messages.calls[0]
    assert "system" not in call


def test_anthropic_joins_text_blocks_and_maps_usage():
    llm = anthropic_provider(["你好"])
    llm.generate_text(messages(user="hi"))

    assert llm.last_call.provider == "anthropic"
    assert llm.last_call.usage.prompt_tokens == 7          # input_tokens
    assert llm.last_call.usage.completion_tokens == 3      # output_tokens


def test_anthropic_structured_output_uses_same_pipeline():
    payload = {"current_step": "建模", "has_obvious_error": False}
    llm = anthropic_provider([json.dumps(payload, ensure_ascii=False)])

    result = llm.generate_structured(messages("s", "u"), StepAnalysis)

    assert result.current_step == "建模"
    assert "JSON Schema" in llm._client.messages.calls[0]["system"]


def test_anthropic_retries_like_openai_provider():
    llm = anthropic_provider([FakeAPITimeoutError("慢"), "成功"], max_retries=1)
    assert llm.generate_text(messages(user="x")) == "成功"
    assert len(llm._client.messages.calls) == 2


# ---------------------------------------------------------------- 消息工具

def test_messages_helper_omits_empty_parts():
    assert messages() == []
    assert messages(system="s") == [{"role": "system", "content": "s"}]
    assert messages(user="u") == [{"role": "user", "content": "u"}]


def test_append_instruction_creates_system_when_absent():
    result = _append_instruction(messages(user="u"), "\n\n要求")
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_append_to_last_user_targets_last_user_message():
    base = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "第一条"},
        {"role": "assistant", "content": "回答"},
        {"role": "user", "content": "第二条"},
    ]
    result = _append_to_last_user(base, " 反馈")

    assert result[-1]["content"] == "第二条 反馈"
    assert result[1]["content"] == "第一条"          # 前面的 user 消息不受影响
    assert base[1]["content"] == "第一条"            # 原列表未被就地修改


# ---------------------------------------------------------------- 工厂

def test_factory_openai_compatible_gateway(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-gateway")
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.local/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")

    llm = get_llm()

    assert isinstance(llm, OpenAICompatibleProvider)
    assert llm.model == "some-model"
    assert llm.base_url == "https://gateway.local/v1"


def test_factory_openai_compatible_requires_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-gateway")
    monkeypatch.setenv("LLM_MODEL", "")

    with pytest.raises(LLMConfigError, match="没有默认模型"):
        get_llm()


def test_factory_role_specific_model(monkeypatch):
    """P17：按节点换模型（便宜模型做匹配、强模型做讲解）。"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("TUTOR_MODEL", "deepseek-reasoner")
    # Explicitly exercise the unconfigured role, independent of a user's demo .env.
    monkeypatch.setenv('STEP_MATCHER_MODEL', '')

    assert get_llm().model == "deepseek-chat"
    assert get_llm("tutor").model == "deepseek-reasoner"
    assert get_llm("step_matcher").model == "deepseek-chat"   # 未覆盖则回落


def test_factory_anthropic_provider(monkeypatch):
    """工厂把 anthropic 配置正确接到 AnthropicProvider。

    anthropic SDK 是可选依赖（未安装时选该 provider 会得到明确的 LLMConfigError），
    所以这里替换掉 provider 类本身，只验证工厂的接线与参数。
    """
    captured = {}

    class StubAnthropic(AnthropicProvider):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(client=FakeAnthropic([]), **kwargs)

    monkeypatch.setattr(factory_module, "AnthropicProvider", StubAnthropic)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://claude.local")
    monkeypatch.setenv("LLM_MODEL", "")

    llm = get_llm()

    assert isinstance(llm, AnthropicProvider)
    assert llm.model == "claude-3-5-sonnet-latest"
    assert llm.describe().kind == "anthropic"
    assert captured["api_key"] == "sk-ant"
    assert captured["base_url"] == "https://claude.local"


def test_anthropic_without_sdk_raises_actionable_error(monkeypatch):
    """没装 anthropic SDK 时，必须给出可操作的提示而不是 ImportError。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LLMConfigError, match="pip install anthropic"):
        AnthropicProvider(api_key="k", model="claude-test")


def test_factory_caches_by_resolved_config(monkeypatch):
    """配置不变时复用实例（连接池生效）；配置一变就换新实例。"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "m1")
    first = get_llm()

    assert get_llm() is first                      # 同配置复用

    monkeypatch.setenv("LLM_MODEL", "m2")
    get_settings.cache_clear()
    second = get_llm()

    assert second is not first
    assert second.model == "m2"


def test_describe_providers_hides_keys(monkeypatch):
    from app.services.llm import describe_providers

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    get_settings.cache_clear()

    overview = describe_providers()

    assert overview["deepseek"]["configured"] is True
    assert "sk-secret" not in json.dumps(overview)     # 不泄露密钥
    assert "anthropic" in overview
    assert "openai_compatible" in overview
