"""LLM provider 工厂与注册表（P17）。

切换厂商只改 `.env`，代码零改动：

    LLM_PROVIDER=deepseek            # DeepSeek 官方
    LLM_PROVIDER=qwen                # 阿里百炼
    LLM_PROVIDER=openai_compatible   # 任意 OpenAI 兼容网关（配 LLM_API_KEY / LLM_BASE_URL）
    LLM_PROVIDER=anthropic           # Claude

新增一家兼容网关：在 `_PROVIDERS` 里加一行即可，无需改上层代码。
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.services.llm.anthropic_provider import AnthropicProvider
from app.services.llm.base import LLMConfigError, LLMProvider
from app.services.llm.openai_compatible import OpenAICompatibleProvider
from app.services.llm.circuit import attach_guard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    """一家 provider 的接入信息。"""

    kind: str                    # openai_compatible / anthropic
    key_field: str               # Settings 里的 API Key 字段
    url_field: str               # Settings 里的 base_url 字段
    default_model: str = ""      # 未显式配置模型时的默认值
    extra_key_field: str = ""    # 备用 Key 字段（如通用 LLM_API_KEY）


_PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        kind="openai_compatible",
        key_field="deepseek_api_key",
        url_field="deepseek_base_url",
        default_model="deepseek-chat",
    ),
    "qwen": ProviderSpec(
        kind="openai_compatible",
        key_field="qwen_api_key",
        url_field="qwen_base_url",
        default_model="qwen-plus",
    ),
    # 任务书 P17 建议的通用入口：任意 OpenAI 兼容服务
    "openai_compatible": ProviderSpec(
        kind="openai_compatible",
        key_field="llm_api_key",
        url_field="llm_base_url",
        default_model="",
        extra_key_field="openai_api_key",
    ),
    "openai": ProviderSpec(
        kind="openai_compatible",
        key_field="llm_api_key",
        url_field="llm_base_url",
        default_model="gpt-4o-mini",
        extra_key_field="openai_api_key",
    ),
    "anthropic": ProviderSpec(
        kind="anthropic",
        key_field="anthropic_api_key",
        url_field="anthropic_base_url",
        default_model="claude-3-5-sonnet-latest",
    ),
}

SUPPORTED_PROVIDERS = tuple(sorted(_PROVIDERS))


def _role_model_field(role: str) -> str:
    """角色 → Settings 里的模型覆盖字段名，如 step_matcher → step_matcher_model。"""
    return f"{role.strip().lower()}_model"


def _resolve_model(settings, role: str, default_model: str) -> str:
    """模型优先级：节点专属配置 > LLM_MODEL > provider 默认。"""
    if role:
        override = (getattr(settings, _role_model_field(role), "") or "").strip()
        if role == 'review' and not override:
            override = (settings.diagnosis_model or '').strip()
        if role == 'rerank' and not override:
            override = (settings.tutor_model or '').strip()
        if override:
            return override
    return (settings.llm_model or "").strip() or default_model


@lru_cache(maxsize=16)
def _build_cached(
    kind: str,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    thinking: str = '',
    max_tokens: int = 0,
) -> LLMProvider:
    """按**解析后的配置**缓存 provider 实例。

    缓存键包含全部生效配置，因此改了 `.env` / 环境变量后自然命中新键、
    拿到新实例——不会像按 (provider, role) 缓存那样读到上一次配置的残留
    （那正是 `tests/test_llm.py` 里工厂用例会踩的坑）。
    配置不变时复用实例，连接池与 keep-alive 才能生效。
    """
    if kind == "anthropic":
        return attach_guard(AnthropicProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        ), provider, base_url, api_key)
    return attach_guard(OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        name=provider,
        thinking=thinking,
        max_tokens=max_tokens,
    ), provider, base_url, api_key)


def _build(provider: str, role: str) -> LLMProvider:
    settings = get_settings()
    spec = _PROVIDERS[provider]

    api_key = (getattr(settings, spec.key_field, "") or "").strip()
    if not api_key and spec.extra_key_field:
        api_key = (getattr(settings, spec.extra_key_field, "") or "").strip()
    if not api_key:
        fields = " 或 ".join(
            field.upper() for field in (spec.key_field, spec.extra_key_field) if field
        )
        raise LLMConfigError(f"缺少 API Key：请在项目根目录 .env 中设置 {fields}")

    model = _resolve_model(settings, role, spec.default_model)
    if not model:
        raise LLMConfigError(
            f"provider {provider!r} 没有默认模型，请在 .env 中设置 LLM_MODEL"
            + (f" 或 {_role_model_field(role).upper()}" if role else "")
        )

    from app.services.request_execution import role_options
    options = {**settings.llm_role_options.get(role, {}), **role_options(role)}
    if set(options) - {'thinking', 'max_tokens'}:
        raise LLMConfigError('不支持的角色调用选项')
    thinking = options.get('thinking', '')
    max_tokens = options.get('max_tokens', 0)
    if thinking not in ('', 'enabled', 'disabled') or not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise LLMConfigError('角色thinking或max_tokens配置不合法')
    if thinking and provider != 'deepseek':
        raise LLMConfigError('thinking选项当前仅支持DeepSeek，不能静默转发给其他供应商')
    return _build_cached(
        spec.kind,
        provider,
        model,
        (getattr(settings, spec.url_field, "") or "").strip(),
        api_key,
        settings.llm_temperature,
        settings.graph_generator_timeout if role == 'graph_generator' else
        (settings.review_timeout if role == 'review' else settings.llm_timeout),
        settings.review_max_retries if role == 'review' else settings.llm_max_retries,
        thinking,
        max_tokens,
    )


def get_llm(role: str = "") -> LLMProvider:
    """按 `LLM_PROVIDER` 返回客户端；`role` 用于按节点覆盖模型。

    role 取值与提示词角色一致：step_matcher / analyze_step / diagnosis / tutor
    （以及将来的 graph_generator）。对应环境变量形如 `STEP_MATCHER_MODEL`。
    """
    settings = get_settings()
    provider = (settings.llm_provider or "").strip().lower()
    if provider not in _PROVIDERS:
        raise LLMConfigError(
            f"未知 LLM_PROVIDER={provider!r}，支持: {list(SUPPORTED_PROVIDERS)}"
        )
    return _build(provider, role.strip().lower())


def clear_cache() -> None:
    """清空 provider 缓存（测试切换配置后调用）。"""
    _build_cached.cache_clear()


def describe_providers() -> dict[str, dict]:
    """各 provider 的当前配置概览（不包含密钥），供调试与 /health 展示。"""
    settings = get_settings()
    overview: dict[str, dict] = {}
    for name, spec in _PROVIDERS.items():
        key = (getattr(settings, spec.key_field, "") or "").strip() or (
            (getattr(settings, spec.extra_key_field, "") or "").strip()
            if spec.extra_key_field
            else ""
        )
        overview[name] = {
            "kind": spec.kind,
            "base_url": (getattr(settings, spec.url_field, "") or "").strip(),
            "default_model": spec.default_model,
            "configured": bool(key),
        }
    return overview


__all__ = [
    "ProviderSpec",
    "SUPPORTED_PROVIDERS",
    "clear_cache",
    "describe_providers",
    "get_llm",
]
