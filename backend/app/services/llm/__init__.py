"""统一 LLM 接口（P17 起为包结构）。

上层代码只依赖本模块导出的抽象，**不出现任何厂商 SDK 对象**：

    from app.services.llm import get_llm
    llm = get_llm()                      # 或 get_llm("diagnosis") 按节点换模型
    text = llm.generate_text(messages)
    result = llm.generate_structured(messages, StepMatch)

模块划分：
- `schemas.py`            LLMMessage / TokenUsage / LLMCall / ProviderInfo
- `base.py`               抽象基类：错误分类、重试、超时、结构化输出
- `openai_compatible.py`  DeepSeek / Qwen / OpenAI / 任意兼容网关
- `anthropic_provider.py` Claude
- `factory.py`            provider 注册表与 `get_llm()`

保持兼容的旧名字：`LLMConfigError` / `LLMStructuredOutputError` /
`OpenAICompatibleClient` / `_extract_json` / `get_llm` 都仍可从本模块导入
（`app.services.llm` 的导入路径不变）。
"""
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import (
    RETRYABLE_ERRORS,
    LLMConfigError,
    LLMError,
    LLMNetworkError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponseError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    classify_exception,
    extract_json,
    is_retryable,
    usage_from_response,
)
from app.services.llm.factory import (
    SUPPORTED_PROVIDERS,
    clear_cache,
    describe_providers,
    get_llm,
)
from app.services.llm.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleProvider,
)
from app.services.llm.schemas import (
    LLMCall,
    LLMMessage,
    ProviderInfo,
    Role,
    TokenUsage,
    messages,
)

# 旧名（Phase 2 起使用，测试与既有代码可能仍在引用）
# - `_extract_json`：原来的私有容错 JSON 提取函数
# - `LLMClient`：原来的 Protocol 名，现在是 LLMProvider 基类的别名
_extract_json = extract_json
LLMClient = LLMProvider

__all__ = [
    # 异常
    "LLMError",
    "LLMConfigError",
    "LLMStructuredOutputError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMNetworkError",
    "LLMResponseError",
    "RETRYABLE_ERRORS",
    "classify_exception",
    "is_retryable",
    # provider
    "LLMProvider",
    "LLMClient",
    "OpenAICompatibleProvider",
    "OpenAICompatibleClient",
    "AnthropicProvider",
    "get_llm",
    "clear_cache",
    "describe_providers",
    "SUPPORTED_PROVIDERS",
    # 数据
    "LLMMessage",
    "LLMCall",
    "ProviderInfo",
    "TokenUsage",
    "Role",
    "messages",
    "extract_json",
    "_extract_json",
    "usage_from_response",
]
