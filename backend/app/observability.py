"""请求级可观测性基础设施（P19）。

任务书 P19 要求每一次 Agent Run 都产生完整 Trace，出问题时能判断是哪一环错了。
其中 `request_id` 需要"贯穿一次 HTTP 请求"——这里用 ContextVar 承载，
由纯 ASGI 中间件在请求进入时写入。

为什么用纯 ASGI 中间件而不是 `BaseHTTPMiddleware`：
后者把下游调用包在额外的任务里，ContextVar 的传播在多版本 Starlette 上行为不一致；
直接写 ASGI 中间件能在同一个上下文里 set/reset，同步端点经
`run_in_threadpool` 执行时会复制该上下文，因此端点里读得到。
"""
import logging
import uuid
from contextvars import ContextVar

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_request_id: ContextVar[str] = ContextVar("dsh_request_id", default="")


def new_request_id() -> str:
    """生成请求 id（16 位十六进制，与 attempt_id 风格一致）。"""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    """当前请求的 id；不在请求上下文中时返回空串。"""
    return _request_id.get()


def set_request_id(value: str) -> object:
    """设置当前上下文的请求 id，返回用于恢复的 token。"""
    return _request_id.set(value)


def reset_request_id(token: object) -> None:
    _request_id.reset(token)  # type: ignore[arg-type]


class RequestIdMiddleware:
    """为每个 HTTP 请求分配（或沿用调用方传入的）request id，并在响应头回显。

    纯 ASGI 实现，见模块 docstring 的说明。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        incoming = ""
        for key, value in scope.get("headers") or []:
            if key == b"x-request-id":
                incoming = value.decode("latin-1").strip()
                break
        request_id = incoming or new_request_id()

        token = set_request_id(request_id)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            reset_request_id(token)
