"""一轮请求内的 LLM 用量归集（P19）。

为什么需要它：`LLMProvider.last_call` 只能看到**最后一次**调用，而一轮分析最多会调用
4 次 LLM（step_matcher / analyze_step / diagnosis / tutor）。Trace 要记录本轮
token 用量与用到的模型，就必须把这一轮的所有调用都收集起来。

用 ContextVar 承载而不是实例属性：FastAPI 同步端点跑在线程池里，
同一次请求内的所有节点共享同一份上下文，并发请求之间互不干扰。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter

from app.services.llm.schemas import LLMCall, TokenUsage

_collector: ContextVar["UsageCollector | None"] = ContextVar("llm_usage_collector", default=None)


@dataclass
class UsageCollector:
    """收集一次请求内的全部 LLM 调用。"""

    calls: list[LLMCall] = field(default_factory=list)
    node_timings_ms: dict[str, float] = field(default_factory=dict)
    request_attempts: int = 0
    failed_attempts: list[str] = field(default_factory=list)

    def add(self, call: LLMCall) -> None:
        self.calls.append(call)

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for call in self.calls:
            total = total + call.usage
        return total

    @property
    def latency_ms(self) -> int:
        return sum(call.latency_ms for call in self.calls)

    @property
    def models(self) -> list[str]:
        """本轮用到的模型（去重、保持首次出现顺序）。"""
        seen: list[str] = []
        for call in self.calls:
            if call.model and call.model not in seen:
                seen.append(call.model)
        return seen

    @property
    def providers(self) -> list[str]:
        seen: list[str] = []
        for call in self.calls:
            if call.provider and call.provider not in seen:
                seen.append(call.provider)
        return seen

    @property
    def retried_calls(self) -> int:
        """发生过网络重试的调用数（排障时先看这个）。"""
        return sum(1 for call in self.calls if call.attempts > 1)

    def as_dict(self) -> dict:
        usage = self.total_usage
        return {
            "calls": len(self.calls),
            "retried_calls": self.retried_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "llm_latency_ms": self.latency_ms,
            "models": self.models,
            "providers": self.providers,
            "node_timings_ms": {k: round(v, 3) for k, v in self.node_timings_ms.items()},
            "actual_attempts": self.request_attempts,
            "failed_attempts": list(self.failed_attempts),
            "reasoning_tokens": usage.reasoning_tokens,
        }

    def model_label(self) -> str:
        """给 `analysis_traces.model` 列用的单值标签。

        单模型（默认情况）就是模型名；按节点配了多个模型时用 `+` 连接，
        既能 grep 也能用于分组统计，完整列表在 token_usage.models 里。
        """
        return "+".join(self.models)


def current_collector() -> UsageCollector | None:
    return _collector.get()


def timed_node(name, fn):
    @wraps(fn)
    def measured(*args, **kwargs):
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            collector = current_collector()
            if collector is not None:
                collector.node_timings_ms[name] = collector.node_timings_ms.get(name, 0) + (perf_counter() - start) * 1000
    return measured


@contextmanager
def track_usage():
    """在上下文内归集所有 LLM 调用；退出时恢复外层状态（支持嵌套）。"""
    collector = UsageCollector()
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)
