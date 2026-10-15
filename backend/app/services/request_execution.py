"""Cooperative request budget and scoped role settings; never cancel a committing thread."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_deadline = ContextVar('model_request_deadline', default=None)
_roles = ContextVar('request_role_options', default=None)


def remaining_seconds():
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def role_options(role):
    return dict((_roles.get() or {}).get(role, {}))


def bounded_timeout(default):
    remaining = remaining_seconds()
    if remaining is None:
        return default
    if remaining <= 0:
        from app.services.llm.base import LLMTimeoutError
        raise LLMTimeoutError('本次照片分析的时间预算已用完；已识别原文保留，请查看结果后决定下一步')
    return min(default, remaining)


@contextmanager
def request_execution(seconds=0, *, roles=None):
    old = _deadline.get()
    proposed = time.monotonic() + seconds if seconds > 0 else None
    deadline = min(old, proposed) if old is not None and proposed is not None else old or proposed
    token = _deadline.set(deadline)
    rtoken = _roles.set(roles if roles is not None else _roles.get())
    try:
        yield
    finally:
        _roles.reset(rtoken)
        _deadline.reset(token)
