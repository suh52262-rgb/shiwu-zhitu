"""OpenAI 兼容 provider（DeepSeek / Qwen / OpenAI / 其它兼容网关）。

覆盖绝大多数国内可用服务：它们都提供 `/chat/completions` 兼容端点，
因此只需换 base_url + api_key + model。

`OpenAICompatibleClient` 是旧名字，保留为别名，避免既有 import 失效。
"""
from typing import Any

from app.services.llm.base import LLMConfigError, LLMProvider, usage_from_response
from app.services.llm.schemas import LLMMessage, TokenUsage


class OpenAICompatibleProvider(LLMProvider):
    """走 openai SDK 的通用 provider。

    client 参数仅供测试注入假 SDK，生产代码不传。
    """

    kind = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
        name: str = "openai_compatible",
        client: Any | None = None,
        thinking: str = '',
        max_tokens: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.name = name
        self.thinking = thinking
        self.max_tokens = max_tokens
        if client is None:
            from openai import OpenAI  # 延迟导入，未配置 Key 时不必加载 SDK

            # timeout 必须显式传入：SDK 默认超时很长，配置错误时会长时间挂住。
            # max_retries=0 也必须显式关掉：SDK 自己默认还重试 2 次，
            # 与基类 _with_retry 叠加后最坏发 3×3=9 次请求——既让
            # LLMCall.attempts 低报真实请求数，也会把一轮分析拖到几分钟。
            # 重试策略统一归基类（只有它知道哪些错误值得重试）。
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._client = client

    def _complete(self, messages: list[LLMMessage]) -> tuple[str, TokenUsage]:
        options = {}
        from app.services.request_execution import remaining_seconds, bounded_timeout
        if remaining_seconds() is not None:
            options['timeout'] = bounded_timeout(self.timeout)
        if self.thinking:
            options['extra_body'] = {'thinking': {'type': self.thinking}}
        if self.max_tokens:
            options['max_tokens'] = self.max_tokens
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=list(messages),
            **options,
        )
        choices = getattr(resp, "choices", None) or []
        if not choices:
            # 网关"200 但空 choices"（错误体伪装成功 / 内容被过滤）时，
            # 直接下标访问会抛 IndexError，被兜底成 LLMResponseError 且丢掉真实原因
            from app.services.llm.base import LLMResponseError

            raise LLMResponseError(f"模型返回了空 choices：{getattr(resp, 'error', None)}")
        choice = choices[0]
        if getattr(choice, 'finish_reason', '') == 'length':
            from app.services.llm.base import LLMResponseError
            raise LLMResponseError('模型输出达到长度限制，不能把截断结果当完整判断')
        text = (choice.message.content or "") if getattr(choice, "message", None) else ""
        return text, usage_from_response(getattr(resp, "usage", None))


# 旧类名别名（Phase 2 起的对外名字，测试与外部脚本可能仍在用）
OpenAICompatibleClient = OpenAICompatibleProvider


__all__ = ["OpenAICompatibleProvider", "OpenAICompatibleClient", "LLMConfigError"]
