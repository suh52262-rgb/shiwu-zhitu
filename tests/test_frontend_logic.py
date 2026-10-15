"""学生端的纯逻辑测试（`frontend/ui_logic.py`）。

## 为什么加这一层
学生端 336 行，**原来零测试覆盖**。已经出过一次真 bug：
新学生（404）被显示成"画像暂不可用（**数据库未连接**）"，
而且那条"新学生"的分支**是死代码**（没调 `raise_for_status()`，httpx 不抛）。

**867 个测试全在 backend** —— 这类"契约变了、界面不报错、只显示一句错话"
的问题，没有任何东西拦得住。

## 做法
把"后端响应 → 界面该显示什么"抽成**不依赖 streamlit** 的纯函数，
在 `ui_logic.py` 里，然后在这里测。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'frontend'))))

from ui_logic import (  # noqa: E402
    MODE_ERROR,
    MODE_NEW,
    MODE_OK,
    connection_error_text,
    health_view,
    problems_from_payload,
    profile_view,
)


# ================================================================ 题目列表

def test_problems_reads_the_items_wrapper():
    """**后端返回 `{"items": [...], "total": N}`，不是裸数组。**

    这个形状写错过一次（核对端点时探针解析错了，误判"demo 五道不在题库里"）。
    """
    got = problems_from_payload(200, {"items": [
        {"problem_id": "demo_01", "title": "吊扇"},
        {"problem_id": "phy_001", "title": "斜面"},
    ], "total": 2})
    assert set(got) == {"demo_01", "phy_001"}
    assert got["demo_01"]["title"] == "吊扇"


def test_problems_accepts_bare_id_field():
    """兼容 `id` 字段（题库接口两种写法都用过）。"""
    got = problems_from_payload(200, {"items": [{"id": "x1"}]})
    assert got == {"x1": {"id": "x1"}}


@pytest.mark.parametrize("status,payload", [
    (500, {"items": [{"problem_id": "a"}]}),
    (200, {}),                       # 没有 items
    (200, {"items": []}),            # 空题库
    (200, {"items": "不是数组"}),
    (200, None),
    (200, [{"problem_id": "a"}]),    # 裸数组（旧形状）
])
def test_problems_unusable_returns_none(status, payload):
    """**用不了就返回 `None`，让调用方回退扫本地。**

    注意空题库也返回 `None` —— 不能拿一个空的后端响应
    把本地那份好好的题库盖掉。
    """
    assert problems_from_payload(status, payload) is None


def test_problems_skips_broken_entries():
    """单条坏数据不拖垮整个列表。"""
    got = problems_from_payload(200, {"items": [
        {"problem_id": "good"}, {"title": "没有 id"}, "不是字典",
    ]})
    assert got == {"good": {"problem_id": "good"}}


# ================================================================ 画像

def test_profile_404_is_a_new_student_not_an_error():
    """**核心：404 是"新学生"，不是"数据库未连接"。**

    这是每个学生的第一个状态，原来被显示成后者 —— **数据库好得很。**
    """
    v = profile_view(404, {"detail": "学生 x 不存在"})
    assert v["mode"] == MODE_NEW
    assert "新学生" in v["caption"]
    assert "数据库" not in v["caption"], "又把新学生说成数据库问题了"
    assert "不可用" not in v["caption"]


def test_profile_other_status_shows_the_code():
    """非 200 要把**状态码显出来**，便于排查；不要笼统说"不可用"。"""
    v = profile_view(503, {})
    assert v["mode"] == MODE_ERROR
    assert "503" in v["caption"]


def test_profile_ok_counts_attempts():
    v = profile_view(200, {
        "attempts": [{"final_correct": True}, {"final_correct": False},
                     {"final_correct": True}],
        "weak_concepts": ["历史求助"],
        "profile_policy_version": "paper_evidence_v1",
        "observed_error_concepts": ["动量定理"],
    })
    assert v["mode"] == MODE_OK
    assert v["stats"]["attempt_count"] == 3
    assert v["stats"]["done"] == 2
    assert v["stats"]["weak"] == ["动量定理"]


@pytest.mark.parametrize("payload", [
    {},                                   # 两个字段都缺
    {"attempts": None, "weak_concepts": None},
    {"attempts": "不是数组", "weak_concepts": 3},
    {"attempts": [None, "x", {}]},        # 元素是垃圾
])
def test_profile_ok_never_crashes_on_missing_fields(payload):
    """**200 但字段缺失/类型不对不能崩。**

    原来是 `profile["attempts"]` 直接取 —— 字段一少就 KeyError，
    再被外层 `except` 吞掉变成一句错话。
    """
    v = profile_view(200, payload)
    assert v["mode"] == MODE_OK
    assert isinstance(v["stats"]["attempt_count"], int)
    assert isinstance(v["stats"]["done"], int)


def test_profile_200_with_non_dict_payload():
    v = profile_view(200, ["不是字典"])
    assert v["mode"] == MODE_ERROR
    assert "格式异常" in v["caption"]


# ================================================================ 健康检查

def test_health_ok():
    v = health_view(200, {"status": "ok"})
    assert v["ok"] is True and "ok" in v["text"]


@pytest.mark.parametrize("status,payload", [
    (500, {"detail": "boom"}),
    (200, {"detail": "没有 status 字段"}),
    (200, {}),
])
def test_health_not_ok_does_not_claim_connected(status, payload):
    """**非 200 或没有 status 时不能报"已连接"。**

    原来是 `resp.json()['status']` 直接取，非 200 会 KeyError，
    然后被 `except` 说成"后端未连接" —— **其实后端连着、只是返回异常。**
    """
    v = health_view(status, payload)
    assert v["ok"] is False
    assert "已连接（" not in v["text"]


def test_connection_error_text_has_start_command():
    """连不上时要**告诉人怎么启动**，不能只说"未连接"。"""
    t = connection_error_text("http://127.0.0.1:8000", RuntimeError("refused"))
    assert "uvicorn" in t and "127.0.0.1:8000" in t
    assert "refused" in t


# ================================================================ 接线

def _code_only(path: Path) -> str:
    """源码去掉注释与文档字符串，**只留真正会执行的代码**。

    ## 为什么必须这样
    第一版直接对着全文做子串检查，结果**误报**了 ——
    那些"旧写法"的字符串出现在我写的**解释性注释**里
    （注释里引用了旧代码来交代问题）。

    **修法是收紧检查口径，不是删掉那条断言。**
    """
    import io
    import tokenize

    out = []
    with path.open(encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return "\n".join(out)


def test_app_uses_the_pure_logic():
    """**防"抽了模块但界面没用上"** —— 那等于没抽。"""
    src = _code_only(Path(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')))
    for fn in ("problems_from_payload", "profile_view", "health_view"):
        assert fn in src, f"界面没调用 {fn}"
    # 旧写法不许回来
    assert "HTTPStatusError" not in src,\
        "那条死代码又回来了（没 raise_for_status，它永不触发）"
    # 直接下标取字段的写法
    assert '["attempts"]' not in src, "画像又直接下标取 attempts 了"
    assert '["status"]' not in src, "健康检查又直接取 status 了"


def test_legacy_profile_not_shown_as_confirmed_error():
    result = profile_view(200, {"attempts": [], "weak_concepts": ["仅求助"]})
    assert result["stats"]["weak"] == []
