"""测试全局夹具。

P11 之后 StudentState 由 `AttemptStore` 承载，默认以 PostgreSQL 为准。
这对单元测试有两个影响：

1. 用例之间会通过数据库互相污染（上一个用例留下的未结束 attempt 会被下一个续做）；
2. 单测不再是"离线、不发网络请求"的，数据库没起来时每个用例都要等连接超时。

因此这里为每个用例提供一个**独占的 AttemptStore**：默认走纯内存模式
（`AttemptStore(persistent=False)`，完全不触碰数据库），用例因此离线且状态干净。
需要真实数据库的用例用 `@pytest.mark.real_db` 声明。

注意：这里只替换 `get_attempt_store()` 的返回值，**不去 monkeypatch
`app.db.persistence`**——后者是全局副作用，会误伤直接调用持久化层的集成测试。
"""
import os
import socket
import threading
import pytest
from app.api import attempts as attempts_api
from app.db import persistence
from app.services import attempt_store as attempt_store_module
from app.services.attempt_store import AttemptStore


@pytest.fixture(scope='session', autouse=True)
def verify_independent_assets():
    from tests.fixture_assets import verify_fixture_bundle
    verify_fixture_bundle()


@pytest.fixture(autouse=True)
def independent_problem_bank(request, tmp_path_factory, monkeypatch, verify_independent_assets):
    """Tests use test-owned synthetic inputs, never the operator's private question bank."""
    if request.node.get_closest_marker('local_asset') is not None:
        yield
        return
    from tests.fixture_assets import install_bank
    folder = tmp_path_factory.mktemp('synthetic-bank') / 'problems'
    monkeypatch.setenv('PROBLEMS_DIR', str(install_bank(folder)))
    yield


@pytest.fixture(scope="session", autouse=True)
def enforce_offline_database():
    """libpq connects in C and bypasses Python's socket guard."""
    if os.environ.get("TEST_OFFLINE") != "1":
        yield
        return
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    def reject_external_db(dialect, conn_rec, cargs, cparams):
        if dialect.name != "sqlite":
            raise OSError("TEST_OFFLINE forbids external database connections")
    event.listen(Engine, "do_connect", reject_external_db)
    try:
        yield
    finally:
        event.remove(Engine, "do_connect", reject_external_db)


@pytest.fixture(autouse=True)
def enforce_offline_network(monkeypatch):
    """CI offline mode fails accidental socket calls instead of using local services."""
    if os.environ.get("TEST_OFFLINE") == "1":
        internal = threading.local()
        original_pair = socket.socketpair
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        def socketpair(*args, **kwargs):
            # Windows implements socketpair with a private loopback handshake.
            # Permit only calls inside this operation, never general localhost IO.
            internal.socketpair = True
            try:
                return original_pair(*args, **kwargs)
            finally:
                internal.socketpair = False
        def guarded(original):
            def connect(*args, **kwargs):
                if getattr(internal, "socketpair", False):
                    return original(*args, **kwargs)
                raise OSError("TEST_OFFLINE forbids network; inject an external-service fake")
            return connect
        monkeypatch.setattr(socket, "socketpair", socketpair)
        monkeypatch.setattr(socket.socket, "connect", guarded(original_connect))
        monkeypatch.setattr(socket.socket, "connect_ex", guarded(original_connect_ex))


@pytest.fixture(autouse=True)
def clean_db_breaker():
    """每个用例前后都清掉数据库熔断状态。

    **这是必需的，不是保险**：熔断器是**进程全局**的（设计如此——它省下的
    正是"每次调用都付连接超时"），所以一个故意让数据库失败的用例
    （`test_persistence` 里的降级用例）会把熔断器跳开 10 秒，
    紧随其后的真实数据库集成用例就在冷却期内被拦下而失败。

    踩过：跑单个用例通过、跑全量失败 7 个，就是这个原因。
    """
    persistence._breaker_open_until = 0.0
    yield
    persistence._breaker_open_until = 0.0


@pytest.fixture(autouse=True)
def isolated_attempt_store(request, monkeypatch):
    """每个用例独占一个干净的 AttemptStore；默认内存模式，`real_db` 标记则直连数据库。"""
    store = AttemptStore(persistent=request.node.get_closest_marker("real_db") is not None)
    # 覆盖两条引用路径：API 层与 services 层各自 import 了该函数
    monkeypatch.setattr(attempts_api, "get_attempt_store", lambda: store)
    monkeypatch.setattr(attempt_store_module, "get_attempt_store", lambda: store)
    yield store


@pytest.fixture(autouse=True)
def clean_import_runs():
    """收尾测试往**真实库里**写的导入批次。

    测试会调 `start_import_run()` 验证状态语义，这些记录留着就会：
      · 让 `--runs` 一直显示一堆 `xxx-test-doc`
      · 让"可疑批次"的告警失去意义（满屏都是）

    做法是**记录测试前的 id 集合、测试后删掉新增的**——
    比按 doc_id 命名约定过滤可靠（不依赖测试作者起什么名字）。
    """
    if os.environ.get("TEST_OFFLINE") == "1":
        yield
        return
    try:
        from app.rag import block_store

        before = {r["id"] for r in block_store.list_import_runs(limit=1000)}
    except Exception:  # noqa: BLE001 数据库不可用时不影响测试
        yield
        return

    yield

    try:
        from app.rag import block_store

        for r in block_store.list_import_runs(limit=1000):
            if r["id"] not in before:
                block_store.delete_import_run(r["id"])
    except Exception:  # noqa: BLE001 清理失败不该让测试变红
        pass
