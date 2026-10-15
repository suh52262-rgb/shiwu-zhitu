"""统一 LLM 接口测试（Phase 2）。

全部使用假 OpenAI 客户端，不发起任何网络请求；
工厂函数测试通过环境变量注入配置（环境变量优先级高于 .env 文件）。
"""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import get_settings
from app.services.llm import (
    LLMConfigError,
    LLMStructuredOutputError,
    OpenAICompatibleClient,
    _extract_json,
    get_llm,
)


class StepAnalysis(BaseModel):
    """测试用结构化输出 Schema。"""

    current_step: str
    has_obvious_error: bool
    concepts: list[str] = []


# ---------- 假 OpenAI SDK ----------

class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def make_client(responses: list[str]) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="http://fake.local",
        model="test-model",
        client=FakeOpenAI(responses),
    )


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """工厂测试改环境变量后需清掉 get_settings 的 lru_cache。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------- chat ----------

def test_chat_returns_text():
    llm = make_client(["机械能守恒的条件是只有重力或弹力做功。"])
    out = llm.chat("system", "user")
    assert "机械能守恒" in out


def test_chat_sends_system_and_user_messages():
    fake = FakeOpenAI(["ok"])
    llm = OpenAICompatibleClient(
        api_key="k", base_url="http://x", model="m", client=fake
    )
    llm.chat("sys-prompt", "user-prompt")
    messages = fake.chat.completions.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "sys-prompt"}
    assert messages[1] == {"role": "user", "content": "user-prompt"}


# ---------- chat_structured ----------

def test_chat_structured_parses_json_with_fence():
    payload = {
        "current_step": "建立能量方程",
        "has_obvious_error": True,
        "concepts": ["动能定理"],
    }
    llm = make_client(["```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"])
    result = llm.chat_structured("分析作答", "mgh = 1/2mv^2", StepAnalysis)
    assert result.current_step == "建立能量方程"
    assert result.has_obvious_error is True
    assert result.concepts == ["动能定理"]


def test_chat_structured_retries_once_on_invalid_json():
    payload = {
        "current_step": "受力分析",
        "has_obvious_error": False,
        "concepts": ["摩擦力"],
    }
    llm = make_client(["抱歉，我无法输出 JSON。", json.dumps(payload, ensure_ascii=False)])
    result = llm.chat_structured("s", "u", StepAnalysis)
    assert result.current_step == "受力分析"
    # 第二次请求应带上上一次的失败反馈
    calls = llm._client.chat.completions.calls
    assert len(calls) == 2
    assert "无法解析" in calls[1]["messages"][1]["content"]


def test_chat_structured_raises_after_retry_exhausted():
    llm = make_client(["bad output", "still bad"])
    with pytest.raises(LLMStructuredOutputError):
        llm.chat_structured("s", "u", StepAnalysis)


def test_chat_structured_rejects_schema_mismatch():
    # 缺少必填字段 has_obvious_error → 校验失败 → 重试 → 仍失败
    llm = make_client(['{"current_step": "x"}', '{"current_step": "y"}'])
    with pytest.raises(LLMStructuredOutputError):
        llm.chat_structured("s", "u", StepAnalysis)


# ---------- _extract_json ----------

def test_extract_json_handles_plain_and_fenced():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('说明文字 {"a": 1} 后缀') == {"a": 1}
    with pytest.raises(ValueError):
        _extract_json("没有任何 JSON")


# ---------- get_llm 工厂 ----------

def test_factory_deepseek(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "")
    llm = get_llm()
    assert llm.model == "deepseek-chat"
    assert llm.base_url == "https://api.deepseek.com"


def test_factory_qwen_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "")
    llm = get_llm()
    assert llm.model == "qwen-plus"
    assert "dashscope" in llm.base_url


def test_factory_custom_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    assert get_llm().model == "deepseek-reasoner"


def test_factory_missing_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(LLMConfigError):
        get_llm()


def test_factory_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gpt4o")
    with pytest.raises(LLMConfigError):
        get_llm()
