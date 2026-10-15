"""Qdrant 客户端管理（Phase 4）。

双模式：
- 服务模式：QDRANT_URL 可连通（docker compose up 启动）时使用
- 本地模式：连不上服务时自动降级，数据持久化在 data/qdrant_local/，
  无需 Docker，适合开发与 Demo

本地模式的限制（重要）：同一时间只允许**一个进程**打开 `data/qdrant_local/`
（qdrant-client 对该目录取非阻塞独占文件锁）。第二个进程会拿到 RuntimeError。
因此重建索引（`scripts/build_rag.py`）前必须先停掉后端进程。

本模块把"锁被占用"翻译成明确的 `QdrantUnavailable`，调用方据此降级为
"本轮无教材上下文"，而不是把一个 RuntimeError 冒到接口层变成 500。
"""
import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

LOCAL_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_local"


class QdrantUnavailable(RuntimeError):
    """向量库不可用（服务不可达且本地目录被其它进程占用等）。"""


@lru_cache
def get_qdrant() -> QdrantClient:
    """返回 Qdrant 客户端：优先服务模式，失败降级本地模式。

    两种失败都会被翻译成 `QdrantUnavailable`，便于上层统一降级。
    """
    settings = get_settings()
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=2, check_compatibility=False)
        client.get_collections()
        logger.info("Qdrant 服务模式：%s", settings.qdrant_url)
        return client
    except Exception as exc:  # noqa: BLE001 连接失败即降级（服务未启动 / 未装 Docker）
        # 保留原始异常：URL 写错 / TLS 失败 / 服务未启动 是三种不同的故障，
        # 统一吞成一句"不可达"会让排查无从下手
        logger.warning("Qdrant 服务不可达（%s: %s），尝试本地模式", type(exc).__name__, exc)

    try:
        client = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    except Exception as exc:  # noqa: BLE001 最常见的是本地锁被另一个进程占用
        raise QdrantUnavailable(
            f"本地向量库 {LOCAL_QDRANT_PATH} 无法打开（{type(exc).__name__}: {exc}）。"
            "本地模式同一时间只允许一个进程使用；请确认没有另一个后端进程在跑，"
            "或改用 QDRANT_URL 指向 Qdrant 服务。"
        ) from exc
    logger.info("Qdrant 本地模式：%s", LOCAL_QDRANT_PATH)
    return client


def clear_cache() -> None:
    """清空客户端缓存（测试隔离或希望重新探测服务模式时使用）。"""
    get_qdrant.cache_clear()
