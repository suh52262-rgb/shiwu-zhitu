"""LLM Provider 抽象基类：统一错误分类、重试、超时与结构化输出（P17）。

设计要点
--------
1. **上层只依赖本模块**，不出现 `OpenAI(...)` / `Anthropic(...)`。
2. **同步接口**：本项目的 LangGraph 节点、FastAPI 端点、AttemptStore 全是同步的，
   为保持一致性这里不做 async。任务书示例里的 `await` 是示意"统一入口"，
   真要改异步时只需在基类加 `agenerate_*` 并让子类实现 `_acomplete`。
3. **错误分类**：把各厂商 SDK 的异常翻译成下面 6 个领域异常，
   上层（尤其 P20 的错误恢复）才能按语义决策，而不是 try/except Exception。
4. **重试策略**：只重试"值得重试"的（超时 / 限流 / 网络 / 5xx），
   配置错误与 4xx 直接抛——重试一个 401 只会浪费时间。
5. **结构化输出重试与网络重试分开**：前者是"模型没按格式回"，带错误反馈再问一次；
   后者是"请求根本没成功"。两者计数独立，互不干扰。
"""
import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.services.llm.fallback import lenient_parse, safe_default
from app.services.llm.schemas import (
    LLMCall,
    LLMMessage,
    ProviderInfo,
    StructuredOutcome,
    TokenUsage,
    messages as build_messages,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------- 异常

class LLMError(RuntimeError):
    """所有 LLM 相关异常的基类。"""


class LLMConfigError(LLMError):
    """LLM 配置缺失或非法（如未填 API Key、未知 provider）。"""


class LLMStructuredOutputError(LLMError):
    """LLM 多次输出均无法解析为合法结构化数据。"""


class LLMTimeoutError(LLMError):
    """请求超时。"""


class LLMRateLimitError(LLMError):
    """被限流（HTTP 429）。"""


class LLMNetworkError(LLMError):
    """连接失败 / DNS / TLS 等网络层错误。"""


class LLMResponseError(LLMError):
    """服务端返回了错误响应（非 429 的 4xx/5xx）。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# 值得重试的错误（配置错误、schema 错误不在其中）
RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (
    LLMTimeoutError,
    LLMRateLimitError,
    LLMNetworkError,
)


# ---------------------------------------------------------------- 提示词

# 追加到 system 提示词后的结构化输出要求
JSON_INSTRUCTION = (
    "\n\n输出要求：必须只输出一个合法的 JSON 对象"
    "（不要输出解释文字，不要使用代码块围栏），"
    "字段必须符合以下 JSON Schema：\n{schema_json}"
)
RETRY_SUFFIX = (
    "\n\n注意：你上一次的输出无法解析为合法 JSON 或不符合 Schema（{err}）。"
    "请严格按要求只输出一个合法的 JSON 对象。"
)


def extract_json(text: str) -> dict:
    """容错提取 LLM 输出中的 JSON 对象（容忍代码围栏与前后缀文字）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"输出中未找到 JSON 对象: {text[:120]!r}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------- 异常翻译

def classify_exception(exc: BaseException) -> LLMError:
    """把厂商 SDK 异常翻译成领域异常。

    刻意按**结构化特征**（类名 / status_code）判断而不 import 各厂商 SDK 的异常类：
    这样 provider 可以延迟导入 SDK，也不会因为 SDK 版本改名而失效。
    """
    if isinstance(exc, LLMError):
        return exc

    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if not isinstance(status, int):
        status = None

    if status == 429 or "RateLimit" in name:
        return LLMRateLimitError(f"{name}: {exc}")
    if "Timeout" in name:
        return LLMTimeoutError(f"{name}: {exc}")
    if "Connection" in name or "Network" in name:
        return LLMNetworkError(f"{name}: {exc}")
    if status is not None:
        return LLMResponseError(f"{name}: {exc}", status_code=status)
    return LLMResponseError(f"{name}: {exc}")


def is_retryable(error: LLMError) -> bool:
    """是否值得重试：限流/超时/网络错误，或 408/425 与 5xx。

    408 必须算进来：DeepSeek 与不少中转网关在过载时回 408，
    openai SDK 自己也把 408 当作可重试（`_should_retry`）。
    """
    if isinstance(error, RETRYABLE_ERRORS):
        return True
    if isinstance(error, LLMResponseError) and error.status_code:
        return error.status_code in (408, 425) or error.status_code >= 500
    return False


# ---------------------------------------------------------------- 基类

class LLMProvider(ABC):
    """所有 LLM provider 的统一接口。

    子类只需实现 `_complete()`（一次原始请求，返回文本与 token 用量），
    重试、超时控制、结构化输出解析、元信息记录都由本基类完成。
    """

    #: 注册表里的 provider 名（子类覆盖）
    name: str = "unknown"
    #: 实现类型（openai_compatible / anthropic），用于元信息展示
    kind: str = "unknown"

    def __init__(
        self,
        model: str,
        base_url: str = "",
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.request_guard = None  # Factory attaches an account-scoped durable guard.
        # 最近一次调用：按线程存放。FastAPI 的同步端点跑在线程池里，
        # 因此同一次请求内读到的就是本次请求的最后一次调用。
        self._local = threading.local()
        self._totals_lock = threading.Lock()
        self._totals = TokenUsage()

    # -------------------------------------------------- 元信息

    def describe(self) -> ProviderInfo:
        """静态元信息（P19 Trace 与 /health 展示用）。"""
        return ProviderInfo(
            name=self.name,
            kind=self.kind,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @property
    def last_call(self) -> LLMCall | None:
        """本线程最近一次调用（含 model / latency / token 用量）。"""
        return getattr(self._local, "last_call", None)

    def reset_last_call(self) -> None:
        """清空本线程的最近调用记录（每轮请求开始时调用，避免读到上一轮的残留）。"""
        self._local.last_call = None

    def usage_totals(self) -> TokenUsage:
        """进程内累计 token 用量（成本核算用；跨线程汇总）。"""
        with self._totals_lock:
            return self._totals

    def _record(self, call: LLMCall) -> None:
        self._local.last_call = call
        with self._totals_lock:
            self._totals = self._totals + call.usage
        # P19：把本次调用汇入当前请求的归集器（没有归集器时是空操作）
        from app.services.llm.tracking import current_collector

        collector = current_collector()
        if collector is not None:
            collector.add(call)

    # -------------------------------------------------- 上层 API

    def generate_text(self, messages: list[LLMMessage]) -> str:
        """普通文本生成（统一入口）。"""
        started = time.perf_counter()
        text, usage, attempts = self._with_retry(messages)
        self._record(
            LLMCall(
                provider=self.name,
                model=self.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempts,
                usage=usage,
            )
        )
        return text

    def generate_structured(self, messages: list[LLMMessage], schema: type[T]) -> T:
        """结构化输出：强制 JSON 并按 Pydantic Schema 校验，失败带反馈重试一次。

        与网络重试相互独立：网络重试在 `_with_retry` 内，格式重试在这里。
        """
        instruction = JSON_INSTRUCTION.format(
            schema_json=json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        attempt_messages = _append_instruction(messages, instruction)
        started = time.perf_counter()
        total_attempts = 0
        usage = TokenUsage()
        last_err = ""

        for round_index in range(2):  # 首次 + 一次带反馈的重试
            payload = (
                _append_to_last_user(attempt_messages, RETRY_SUFFIX.format(err=last_err))
                if round_index
                else attempt_messages
            )
            text, round_usage, attempts = self._with_retry(payload)
            total_attempts += attempts
            usage = usage + round_usage
            try:
                result = schema.model_validate(extract_json(text))
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:200]
                continue
            self._record(
                LLMCall(
                    provider=self.name,
                    model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=total_attempts,
                    usage=usage,
                )
            )
            return result

        # 解析彻底失败也要记账：请求真发出去了、token 真消耗了，
        # 不记的话最需要排障的那次调用在 trace 与成本核算里完全不存在。
        self._record_failed_call(started, total_attempts, usage)
        raise LLMStructuredOutputError(f"结构化输出解析失败（已重试一次）: {last_err}")

    def _record_failed_call(self, started: float, attempts: int, usage: TokenUsage) -> None:
        """把一次**失败**的调用也记进元信息。

        否则最需要排障的那次调用在 trace / 成本核算里完全不存在：
        两次请求真金白银发出去、token 也消耗了，却什么都看不到。
        """
        self._record(
            LLMCall(
                provider=self.name,
                model=self.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempts,
                usage=usage,
            )
        )

    def chat_structured_resilient(
        self, system: str, user: str, schema: type[T]
    ) -> StructuredOutcome[T]:
        """结构化输出 + 完整降级链（P20），**永不抛异常**。

            严格解析（含一次带反馈重试）
              → 仍失败 → 宽松解析（从脏输出里抢救真实字段）
              → 仍失败 → 安全默认实例
              → 连兜底都失败（不该发生）→ degraded 且 value=None

        调用方**必须**看 `degraded`：拿到的是"能用的值"还是"兜底产物"，
        决定它该不该把这个值写进学生状态。对判断类字段（是否出错、匹配到哪个节点），
        应当选择不采用并只记录 degrade 原因——编造一个"看起来合理"的判断，
        比直接报错更糟（见 fallback.py 的模块说明）。
        """
        messages: list[LLMMessage] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return StructuredOutcome(value=self.generate_structured(messages, schema))
        except (LLMStructuredOutputError, LLMResponseError) as exc:
            # 模型答了，只是格式坏 → 抢救。原始文本拿不到了（generate_structured
            # 内部已消费），所以这里直接进安全默认；宽松解析留给"能拿到文本"的路径。
            return StructuredOutcome(
                value=self._safe_default(schema, str(exc)),
                degraded=True,
                reasons=[f"结构化输出失败（{type(exc).__name__}）：{str(exc)[:160]}"],
            )
        except LLMError as exc:
            # 网络/限流/超时：重试已在 _with_retry 内耗尽
            return StructuredOutcome(
                value=self._safe_default(schema, str(exc)),
                degraded=True,
                reasons=[f"模型调用失败（{type(exc).__name__}）：{str(exc)[:160]}"],
            )
        except Exception as exc:  # noqa: BLE001 节点不该因为任何异常而击穿整轮
            return StructuredOutcome(
                value=self._safe_default(schema, str(exc)),
                degraded=True,
                reasons=[f"未预期错误（{type(exc).__name__}）：{str(exc)[:160]}"],
            )

    @staticmethod
    def _safe_default(schema: type[T], reason: str) -> T | None:
        try:
            return safe_default(schema)
        except Exception:  # noqa: BLE001 兜底本身失败就只能返回 None
            logger.warning("安全默认实例构造失败（%s）：%s", schema.__name__, reason)
            return None

    def chat_structured_lenient(self, system: str, user: str, schema: type[T]) -> StructuredOutcome[T]:
        """与 `chat_structured_resilient` 同一条降级链，但**多一次宽松解析**。

        区别：这里自己发一次原始调用（`generate_text`），因此能拿到模型原文，
        格式坏时可以真正走 fallback parser 抢救字段。代价是少了一次
        `generate_structured` 内部的"带反馈重试"。

        用于那些"字段值本身有价值、值得多花一次解析"的节点。
        """
        instruction = JSON_INSTRUCTION.format(
            schema_json=json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        messages: list[LLMMessage] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            text = self.generate_text(_append_instruction(messages, instruction))
        except Exception as exc:  # noqa: BLE001
            return StructuredOutcome(
                value=self._safe_default(schema, str(exc)),
                degraded=True,
                reasons=[f"模型调用失败（{type(exc).__name__}）：{str(exc)[:160]}"],
            )

        parsed, notes = lenient_parse(text, schema)
        if parsed is not None:
            return StructuredOutcome(value=parsed, degraded=True, reasons=[f"宽松解析：{'；'.join(notes)}"])
        return StructuredOutcome(
            value=self._safe_default(schema, "lenient parse failed"),
            degraded=True,
            reasons=[f"宽松解析失败：{'；'.join(notes)}"],
        )

    # ---- 任务书 P17 列出但当前尚未接入的能力：显式声明而不是悄悄缺失 ----

    def generate_stream(self, messages: list[LLMMessage]):
        """流式输出（未实现）。"""
        raise NotImplementedError(
            "流式输出尚未接入：当前前端是一次性展示提示，"
            "待实时对话或 P25 Demo Playback 需要逐字输出时实现"
        )

    def call_tools(self, messages: list[LLMMessage], tools: list[dict]):
        """模型自主调用工具（未实现）。"""
        raise NotImplementedError(
            "Tool Calling 尚未接入：当前 RAG 由工作流节点按状态调用（P7 的 RAG Tool 化），"
            "不是模型自主选工具；待需要模型自选工具时实现"
        )

    # -------------------------------------------------- 兼容旧接口

    def chat(self, system: str, user: str) -> str:
        """兼容旧调用：单轮对话返回文本。"""
        return self.generate_text(build_messages(system, user))

    def chat_structured(self, system: str, user: str, schema: type[T]) -> T:
        """兼容旧调用：结构化输出。"""
        return self.generate_structured(build_messages(system, user), schema)

    # -------------------------------------------------- 子类实现

    @abstractmethod
    def _complete(self, messages: list[LLMMessage]) -> tuple[str, TokenUsage]:
        """发一次原始请求，返回 (文本, token 用量)。实现方需自行处理超时。"""

    # -------------------------------------------------- 重试

    def _with_retry(self, messages: list[LLMMessage]) -> tuple[str, TokenUsage, int]:
        """按退避策略重试，返回 (文本, 用量, 实际请求次数)。"""
        last_error: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            from app.services.request_execution import bounded_timeout, remaining_seconds
            bounded_timeout(self.timeout)  # Expired rounds must not start another paid request.
            ticket = self.request_guard.before_call() if self.request_guard is not None else None
            from app.services.llm.tracking import current_collector
            collector = current_collector()
            if collector is not None:
                collector.request_attempts += 1
            try:
                text, usage = self._complete(messages)
                return text, usage, attempt + 1
            except Exception as exc:  # noqa: BLE001 统一翻译后再决定是否重试
                error = classify_exception(exc)
                if collector is not None:
                    collector.failed_attempts.append(type(error).__name__)
                if self.request_guard is not None:
                    self.request_guard.failed(error, ticket)
                last_error = error
                if attempt >= self.max_retries or not is_retryable(error):
                    raise error from exc
                delay = _backoff_seconds(attempt)
                remaining = remaining_seconds()
                if remaining is not None and remaining <= delay:
                    raise error from exc
                logger.warning(
                    "LLM 调用失败（第 %d 次，%s），%.1fs 后重试：%s",
                    attempt + 1,
                    type(error).__name__,
                    delay,
                    error,
                )
                time.sleep(delay)
        raise last_error or LLMError("LLM 调用失败")


def _backoff_seconds(attempt: int) -> float:
    """指数退避 + 抖动，避免多个请求同时重试再次打满限流。"""
    return min(2.0**attempt, 8.0) * (0.5 + random.random() / 2)


def _append_instruction(messages: list[LLMMessage], instruction: str) -> list[LLMMessage]:
    """把结构化输出要求追加到 system 消息；没有 system 就补一条。"""
    result = [dict(message) for message in messages]  # type: ignore[misc]
    for message in result:
        if message["role"] == "system":
            message["content"] = message["content"] + instruction
            return result  # type: ignore[return-value]
    return [{"role": "system", "content": instruction.lstrip()}, *result]  # type: ignore[list-item]


def _append_to_last_user(messages: list[LLMMessage], suffix: str) -> list[LLMMessage]:
    """把重试反馈追加到最后一条 user 消息。"""
    result = [dict(message) for message in messages]  # type: ignore[misc]
    for message in reversed(result):
        if message["role"] == "user":
            message["content"] = message["content"] + suffix
            return result  # type: ignore[return-value]
    return [*result, {"role": "user", "content": suffix.lstrip()}]  # type: ignore[list-item]


def usage_from_response(usage: Any) -> TokenUsage:
    """从各厂商 SDK 的 usage 对象抽取 token 用量。

    刻意做成容错的：测试假件、部分网关、以及不同的 SDK 版本给的字段并不一致
    （缺失时返回空用量，绝不能因为读不到 usage 就让整次调用失败）。
    """
    if usage is None:
        return TokenUsage()

    def pick(*names: str) -> int:
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, int):
                return value
            if isinstance(usage, dict) and isinstance(usage.get(name), int):
                return usage[name]
        return 0

    prompt = pick("prompt_tokens", "input_tokens")
    completion = pick("completion_tokens", "output_tokens")
    total = pick("total_tokens") or (prompt + completion)

    # 两个 details 必须**分别**读取：OpenAI 把 cached_tokens 放在
    # prompt_tokens_details、把 reasoning_tokens 放在 completion_tokens_details。
    # 曾经用 `A or B` 只取其中一个对象、再从它身上读两个字段，
    # 结果两者同时存在时 reasoning_tokens 恒为 0（它按输出计费，会低估成本）。
    reasoning = _detail_value(usage, "completion_tokens_details", "reasoning_tokens")
    # Anthropic 把 prompt cache 命中量放在顶层 cache_read_input_tokens
    cached = _detail_value(usage, "prompt_tokens_details", "cached_tokens") or int(
        pick("cache_read_input_tokens") or 0
    )
    if isinstance(usage, dict):
        reasoning = reasoning or int(usage.get("reasoning_tokens", 0) or 0)
        cached = cached or int(usage.get("cached_tokens", 0) or 0)
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


def _detail_value(usage: Any, container: str, field: str) -> int:
    """从 usage 的某个 details 容器里取一个整数字段，缺失一律返回 0。"""
    details = (
        usage.get(container) if isinstance(usage, dict) else getattr(usage, container, None)
    )
    if details is None:
        return 0
    value = details.get(field) if isinstance(details, dict) else getattr(details, field, None)
    return value if isinstance(value, int) else 0
