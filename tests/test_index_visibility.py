"""`partial` 的检索可见性：索引一致就该可见。

## 这个 bug 的代价
原规则「检索端只认 `ready`」是为了防 **`index_pending`**（索引与 PG 可能不一致），
但把 **`partial`** 一起排除了。

而 `partial` 是"导入完成、索引已重建、**只是有页失败**"——索引是一致的。

实测代价：教材-2 下册与教材-5 电磁学**各失败 1 页**，
**963 页内容全校搜不到**。

## 为什么用黑名单而不是白名单
原来那版是白名单（只认 `ready`），于是**每个新状态都默认不可见**——
`partial` 就是这么被误伤的。黑名单让新状态**默认可见**，
要排除必须显式加进去。
"""
import inspect

import pytest

from app.rag import block_store as bs


# ---------------------------------------------------------------- 可见性判定

@pytest.mark.parametrize("status,visible", [
    ("ready", True),
    # **核心断言**：缺页不影响可见性，索引是一致的
    ("partial", True),
    # 索引可能过期/不一致 → 必须排除
    ("index_pending", False),
    ("running", False),
    ("failed", False),
    # 未知状态：**默认可见**（黑名单语义），新增状态不会静默消失
    ("some_future_status", True),
])
def test_visibility_by_status(status, visible):
    assert bs.is_index_visible(status) is visible


def test_partial_is_not_in_invisible_set():
    """`partial` 不得出现在排除名单里——这是本次修复的核心。"""
    assert bs.RUN_PARTIAL not in bs.INDEX_INVISIBLE_STATUSES


def test_blacklist_semantics_new_status_defaults_visible():
    """**黑名单语义**：新状态默认可见。

    白名单会让每个新状态都默认不可见，重演 `partial` 这个 bug。
    """
    assert bs.is_index_visible("brand_new_status") is True


def test_retriever_uses_is_index_visible():
    """**防"写了没接"**：检索端必须真的用这个判定。"""
    from app.rag import retriever

    src = inspect.getsource(retriever)
    assert "is_index_visible" in src, "检索端仍用旧的白名单判定"
    assert "== _bs.INDEX_READY" not in src, "检索端还在只认 ready"


def test_retriever_docstring_explains_partial():
    """说明要留在代码里，否则下一个人会"顺手改回白名单"。"""
    from app.rag import retriever

    src = inspect.getsource(retriever)
    assert "partial" in src and "可见" in src
