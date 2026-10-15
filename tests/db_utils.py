"""测试共享工具：PostgreSQL 可用性探测。

为什么单独抽一个模块并加缓存：

1. `@pytest.mark.skipif(...)` 是在 **pytest 收集阶段**求值的，也就是说这个探测会在
   收集期发起真实网络 IO——数据库不可达时，所有测试都还没开始跑就被它拖住。
2. 此前 `test_persistence.py` / `test_students_api.py` / `test_attempt_api.py`
   各写了一份 `_db_available()`，同一个不可达地址要被重复等待三次。

配合 `app.db.postgres` 里显式设置的 `connect_timeout`，这里最多阻塞一次，且时长有上限。
"""
from functools import lru_cache
import os


@lru_cache(maxsize=1)
def db_available() -> bool:
    """PostgreSQL 可达则返回 True。

    结果被缓存，整个测试会话只真正探测一次；任何异常（连接超时、认证失败、
    驱动缺失等）都视为"不可用"，对应测试走 skip 而不是报错。
    """
    if os.environ.get("TEST_OFFLINE") == "1":
        return False
    try:
        from sqlalchemy import text

        from app.db.postgres import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
