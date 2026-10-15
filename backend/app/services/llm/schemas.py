"""LLM 调用的通用数据结构（P17）。

这些类型是 provider 之间唯一的公共语言：上层只看到 `LLMCall` / `TokenUsage`，
不会碰到任何厂商 SDK 的对象。P19 的 Trace 需要 model / token_usage / latency，
直接取 `LLMCall` 即可。
"""
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

Role = Literal["system", "user", "assistant"]


class LLMMessage(TypedDict):
    """规范化的对话消息（OpenAI 风格，也是本项目内部的统一格式）。

    Anthropic 没有 messages 里的 system 角色，其适配器会把 system 消息
    提取到顶层 system 参数——这个差异被封装在 provider 内部。
    """

    role: Role
    content: str


def messages(system: str = "", user: str = "") -> list[LLMMessage]:
    """便捷构造：把 system / user 两段文本转成规范消息列表。"""
    result: list[LLMMessage] = []
    if system:
        result.append({"role": "system", "content": system})
    if user:
        result.append({"role": "user", "content": user})
    return result


@dataclass
class TokenUsage:
    """一次（或累计）调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Anthropic 用 input/output 命名，额外保留原始字段便于排查
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def is_empty(self) -> bool:
        return not (self.prompt_tokens or self.completion_tokens or self.total_tokens)

    def as_dict(self) -> dict[str, int]:
        data = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens:
            data["cached_tokens"] = self.cached_tokens
        if self.reasoning_tokens:
            data["reasoning_tokens"] = self.reasoning_tokens
        return data


@dataclass
class LLMCall:
    """一次逻辑调用（含重试）的元信息。"""

    provider: str
    model: str
    latency_ms: int = 0
    attempts: int = 1          # 实际发出的请求次数（网络重试后 >1）
    usage: TokenUsage = field(default_factory=TokenUsage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "token_usage": self.usage.as_dict(),
        }


@dataclass
class ProviderInfo:
    """provider 的静态元信息（不含任何一次调用的状态）。"""

    name: str
    kind: str
    model: str
    base_url: str = ""
    temperature: float = 0.2
    timeout: float = 30.0
    max_retries: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class PhysicsFinding(BaseModel):
    """物理一致性检查的一条发现（P15 Graph Validator 的 LLM 输出）。"""

    node: str = Field(default="", description="涉及的节点 id；整题层面的问题留空")
    description: str = Field(default="", description="一句中文说清哪里不对、为什么")
    severity: str = Field(default="warning", description="error（物理上明确错误）或 warning（可疑）")


class PhysicsCheck(BaseModel):
    """物理一致性检查结果。**没发现问题时必须返回 consistent=true、findings=[]**。"""

    consistent: bool = Field(default=True, description="这张图的物理是否站得住")
    findings: list[PhysicsFinding] = Field(default_factory=list)
    notes: str = Field(default="", description="给审校者的附注，可为空")


@dataclass
class StructuredOutcome(Generic[T]):
    """结构化输出 + 降级信息（P20）。

    `degraded=True` 表示**这个值来自降级链而非模型的正常输出**：
    - 可能是宽松解析从脏文本里抢救出来的真实字段；
    - 也可能是最后的"安全默认实例"（字段是零值，没有信息量）。

    调用方必须区分对待：拿到安全默认实例时**不要**把它当成真实判断写进学生状态。
    `reasons` 会进 Trace，用于发现"模型老是输出坏 JSON"这类需要改提示词的问题。
    """

    value: T | None = None
    degraded: bool = False
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # 方便 `if outcome:` 判断"有没有拿到值"
        return self.value is not None
