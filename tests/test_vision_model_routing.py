"""`call_vision(model=...)` 的模型名解析。

## 这个 bug 长什么样（实测）
```
call_vision(img, prompt, model="deepseek-flash")
  → 请求 dashscope/deepseek-flash
  → 404 The model `deepseek-flash` does not exist
```
原来裸模型名**一律按 `dashscope` 解释**。但 `vision_models()` 返回的就是**裸名**——
**调用方从那里拿到名字再传回来必然失败，往返不成立。**

对比：`model="qwen3-vl-30b-a3b-instruct"` 反而成功，
因为它在候选链里本来就属于 dashscope——**错得刚好蒙对**。

## 顺带一个好消息
那次 404 之后**回退链自动落到 qwen3-vl 并成功**。
这是 `VISION_FALLBACKS` 在真实故障下的第一次实战，表现正确。
"""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.services.vision import (  # noqa: E402
    VISION_FALLBACKS,
    Candidate,
    resolve_candidate,
    vision_models,
)


def test_explicit_provider_prefix():
    got = resolve_candidate("zhipu/glm-4.6v")
    assert got == [Candidate("zhipu", "glm-4.6v")]


def test_bare_name_finds_its_provider():
    """**核心**：裸名要在候选链里找到真正的供应商。

    `deepseek-flash` 在候选链里是 `deepseek` 的，不是 `dashscope` 的。
    """
    got = resolve_candidate("deepseek-flash")
    assert got == [Candidate("deepseek", "deepseek-flash")]
    assert got[0].provider != "dashscope", "又按 dashscope 解释了"


def test_round_trip_from_vision_models():
    """**往返**：`vision_models()` 给的裸名传回来必须能解析对。

    这正是原 bug 的形态——调用方从 `vision_models()` 取名再传回，
    结果被当成 dashscope 的模型，404。
    """
    for name in vision_models():
        got = resolve_candidate(name)
        assert got, f"{name} 解析不出候选"
        # 解析出的 (provider, model) 必须在已知链里
        assert got[0] in VISION_FALLBACKS, (
            f"{name} 被解析成 {got[0]}，不在候选链里——往返断了"
        )


def test_unknown_bare_name_falls_back_to_dashscope():
    """未知裸名仍按 dashscope 解释——保持历史行为。"""
    assert resolve_candidate("some-future-model") == [
        Candidate("dashscope", "some-future-model")]


def test_empty_means_full_chain():
    """空字符串 = 不给覆盖，用完整候选链。"""
    assert resolve_candidate("") == list(dict.fromkeys(VISION_FALLBACKS))
    assert resolve_candidate("   ") == list(dict.fromkeys(VISION_FALLBACKS))


def test_call_vision_uses_the_resolver():
    """**防"写了没接"**。"""
    import inspect

    from app.services import vision as v

    src = inspect.getsource(v.call_vision)
    assert "resolve_candidate(model)" in src
    assert 'Candidate("dashscope", model)' not in src, "旧的裸名逻辑还在"
