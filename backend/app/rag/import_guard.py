"""导入的三道护栏：doc_id 安全、并发互斥、图片引用有效。

## 为什么需要

`doc_id` 会被直接拼进输出目录（`data/documents/{doc_id}/figures/`）。
不做校验的话，`--doc-id ../../..` 这类输入能写到仓库外——
`ProblemRepository` 早就有 `_is_safe_id` 处理同类问题（题库 id 也拼路径），
导入这条路当时漏了。

并发导入同一文档时，两个进程会互相覆盖图片、交错写 PG 与 Qdrant，
最后得到一个两个存储都不自洽的状态。

图片路径写进 PG 后如果文件不存在，检索命中时前端会拿到一个死链——
而这在"导入成功"的表面上完全看不出来。
"""
from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.postgres import get_engine

logger = logging.getLogger(__name__)

#: 只允许字母、数字、下划线、短横线、点（点不能开头，见 validate_doc_id）
SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")


class ImportGuardError(RuntimeError):
    """导入被护栏拦下。"""


def validate_doc_id(doc_id: str) -> str:
    """校验 doc_id，返回它本身；不合法则抛异常。

    **必须校验**：这个值会被拼进 `data/documents/{doc_id}/figures/`。
    `../../etc` 这样的输入能写到仓库外。

    不允许以点开头（挡掉 `.` / `..` / 隐藏目录），
    也不允许路径分隔符与冒号（Windows 盘符/ADS）。
    """
    if not doc_id or not isinstance(doc_id, str):
        raise ImportGuardError("doc_id 不能为空")
    if not SAFE_DOC_ID.match(doc_id):
        raise ImportGuardError(
            f"doc_id 不合法：{doc_id!r}。只允许字母、数字、下划线、短横线与点，"
            "且不能以点开头（禁止路径分隔符与目录穿越）。"
        )
    if ".." in doc_id:
        raise ImportGuardError(f"doc_id 不允许包含连续点：{doc_id!r}")
    return doc_id


def safe_join(root: Path, doc_id: str) -> Path:
    """把 doc_id 拼到 root 下，并**确认结果仍在 root 内**。

    双重保险：即使 `validate_doc_id` 将来被放宽或绕过，
    这里 resolve 之后的路径检查仍能挡住目录穿越。
    """
    validate_doc_id(doc_id)
    root = Path(root).resolve()
    candidate = (root / doc_id).resolve()
    if candidate != root and root not in candidate.parents:
        raise ImportGuardError(f"doc_id 会导致路径逃出根目录：{doc_id!r}")
    return candidate


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """算文件摘要。

    用于区分"**同一文件重试**"与"**同名但内容不同的文件**"——
    只看文件名的话，后者会被当成前者，用旧结果糊弄过去。
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- 并发互斥

#: 导入锁的 advisory lock 命名空间（任意常数，只要全项目一致）
_LOCK_NAMESPACE = 0x51_57_55       # "QWU"
_LOCK_ACQUIRED = "__shiwu_import_lock_acquired__"


@contextmanager
def import_lock(doc_id: str):
    """同一 doc_id 同时只允许一个导入在执行。

    用 PostgreSQL 的 **session 级 advisory lock**：
      · 进程崩了连接断开，锁自动释放，不会留下死锁
      · 跨进程有效（`threading.Lock` 只在一个进程内有效）

    **数据库不可用时降级为不锁**（只记日志）——导入本身在
    PG 掉线时会失败在别处，不该因为"锁拿不到"而给出误导性的错误。
    """
    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法获取导入锁（数据库不可用），本次不锁：%s", type(exc).__name__)
        yield False
        return

    key = f"{_LOCK_NAMESPACE}:{doc_id}"
    # pg_try_advisory_lock 用两个 int4 拼 key，这里把字符串摘要成两个整数
    digest = hashlib.sha256(key.encode()).digest()
    k1 = int.from_bytes(digest[:4], "big", signed=True)
    k2 = int.from_bytes(digest[4:8], "big", signed=True)

    with Session(engine) as session:
        got = session.execute(
            text("SELECT pg_try_advisory_lock(:k1, :k2)"), {"k1": k1, "k2": k2}
        ).scalar()
        if not got:
            raise ImportGuardError(
                f"已有另一个导入正在处理 {doc_id}。"
                "同一文档的并发导入会交错写 PG 与 Qdrant，导致两个存储都不自洽——"
                "等它结束，或用 --reconcile 检查当前状态。"
            )
        try:
            yield True
        finally:
            session.execute(
                text("SELECT pg_advisory_unlock(:k1, :k2)"), {"k1": k1, "k2": k2}
            )
            session.commit()


# ---------------------------------------------------------------- 图片校验

def verify_figure(path: str | Path, root: Path) -> tuple[str, str]:
    """校验插图引用。返回 `(规范化后的相对路径, 问题描述)`。

    在两个地方调用：
      · **落库前** —— 文件不存在就不发布该块（`indexable=False`）
      · **对账时** —— 报出所有死链

    第二个返回值非空表示有问题；调用方据此决定是否发布。
    """
    if not path:
        return "", ""
    rel = str(path).replace("\\", "/")
    full = Path(root) / rel
    if not full.is_file():
        return rel, "插图文件不存在"
    if full.stat().st_size == 0:
        return rel, "插图文件为空"
    return rel, ""
