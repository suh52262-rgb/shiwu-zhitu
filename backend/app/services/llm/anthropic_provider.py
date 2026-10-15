"""Anthropic（Claude）provider 适配（P17）。

任务书 P17 要求"Agent 不应绑定某个模型厂商"，因此补上非 OpenAI 协议的实现，
证明抽象层是真的（而不是把 OpenAI 兼容再包一层）。

与 OpenAI 的协议差异都在这里被吸收：
- system 提示词是**顶层参数**，不放在 messages 里；
- 回复在 `content` 的块数组里，需要拼接文本块；
- token 用量字段叫 `input_tokens` / `output_tokens`；
- 必填 `max_tokens`，没有默认值。
"""
from typing import Any

from app.services.llm.base import LLMConfigError, LLMProvider, usage_from_response
from app.services.llm.schemas import LLMMessage, TokenUsage

# Claude 单次回复上限；本项目提示词都很短，4096 足够且能有效控制成本
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """走 anthropic SDK 的 provider。

    client 参数仅供测试注入假 SDK，生产代码不传。
    """

    kind = "anthropic"
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.max_tokens = max_tokens
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMConfigError(
                    "使用 anthropic provider 需要先安装 SDK：pip install anthropic"
                ) from exc
            # max_retries=0：重试统一由基类 _with_retry 负责；叠加 SDK 自带的
            # 2 次重试会变成 9 次请求（同 openai_compatible 的说明）
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
            if base_url:
                kwargs["base_url"] = base_url
            client = Anthropic(**kwargs)
        self._client = client

    def _complete(self, messages: list[LLMMessage]) -> tuple[str, TokenUsage]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]
        if not conversation:  # Anthropic 要求 messages 非空
            conversation = [{"role": "user", "content": ""}]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        resp = self._client.messages.create(**kwargs)
        return _join_text_blocks(resp), usage_from_response(getattr(resp, "usage", None))


def _join_text_blocks(resp: Any) -> str:
    """拼接 content 里的文本块；忽略工具调用等非文本块。"""
    blocks = getattr(resp, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


__all__ = ["AnthropicProvider", "DEFAULT_MAX_TOKENS"]
