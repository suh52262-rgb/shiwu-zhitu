"""PostgreSQL 连接管理（Phase 6 实现建表与会话）。

懒加载 engine：导入不连接，首次使用才建立连接池。
"""
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """SQLAlchemy engine（psycopg3 驱动）。

    connect_timeout 必须显式传入：libpq 默认无限等待，PostgreSQL 未启动时
    engine.connect() 会长时间阻塞（实测 >120s），导致调用方无法及时降级。
    """
    settings = get_settings()
    return create_engine(
        settings.postgres_dsn,
        pool_pre_ping=True,
        connect_args={"connect_timeout": settings.postgres_connect_timeout},
    )
