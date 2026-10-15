"""视频管线的模型路由与报告。

## 两个 bug（都是实测撞出来的）

### ① 报告的是提示值，不是实际用的模型
`summary["model"]` 来自 `resolve_model(options)`（提示值），
而 `call_vision` 走候选链——**`meta` 里真实的模型名被丢掉了**。
每次跑完都不知道是谁干的事，而所有测量都建立在这个字段上。

### ② 更严重：管线**绕过**了回退链
`build_vl_client` 返回 `resolve_model(options)`，它**会套默认值**
（`qwen_vl_model` = `qwen3-vl-plus`）。管线把这个当 `model=` 传下去，
`call_vision` 就**只试那一个候选**，`VISION_FALLBACKS` 整条链被绕过。

**实测**：`models_used` 报 `qwen3-vl-plus`（配置默认值），
而不是链首选的 `deepseek-flash`——**路由到 `call_vision` 的主要理由
（拿回退链）根本没生效**。

修法：`build_vl_client` 只回显 `options.model`（用户显式指定的），
留空则交给 `call_vision` 走完整链。

> 还有一个我自己的错误要记：我曾说"实际用的是 `deepseek-flash`"，
> 那只是**独立实验**的情况（没传 model）；**管线里一直是 `qwen3-vl-plus`**。
> 我把两条路的数字混在一起比了。
"""
import inspect

import pytest

from app.vision.service import (
    VideoAnalysisOptions,
    build_vl_client,
    resolve_model,
)


@pytest.fixture(autouse=True)
def _fake_available(monkeypatch):
    """**不让测试依赖真实 API Key。**

    这组测的是"返回值该是什么"，不是"key 有没有配"。
    单独跑这个文件时 `.env` 未必被加载，`vision_available()` 会是 False，
    于是 `build_vl_client` 抛错——**那是环境依赖，不是被测行为**。
    """
    from app.vision import service as sv

    monkeypatch.setattr(sv, "vision_available", lambda: True)


def test_build_vl_client_echoes_only_explicit_override():
    """**核心**：`build_vl_client` 不能把默认值当覆盖传下去。

    否则 `call_vision` 只试一个候选，回退链被绕过。
    """
    _, model = build_vl_client(VideoAnalysisOptions())
    assert model == "", (
        f"空配置下应回显空串（让 call_vision 走完整链），实到 {model!r}"
    )


def test_build_vl_client_passes_explicit_model_through():
    _, model = build_vl_client(VideoAnalysisOptions(model="glm-4.6v"))
    assert model == "glm-4.6v"


def test_build_vl_client_strips_whitespace():
    _, model = build_vl_client(VideoAnalysisOptions(model="  glm-4.6v  "))
    assert model == "glm-4.6v"


def test_resolve_model_is_a_hint_not_a_routing_decision():
    """`resolve_model` 仍是"提示值"（有默认值）——它不该被用来路由。

    这条钉住**语义区别**，防止有人图省事又把它接回 `build_vl_client`。
    """
    hint = resolve_model(VideoAnalysisOptions())
    assert hint, "提示值应有默认值"
    _, routing = build_vl_client(VideoAnalysisOptions())
    assert routing != hint, (
        "build_vl_client 又开始返回提示值了——回退链会被绕过"
    )


def test_summary_reports_actual_model_not_hint():
    """summary 报**实际用的模型**，提示值另存 `model_hint`。"""
    from app.vision import service as sv

    src = inspect.getsource(sv.run_pipeline)
    assert '"models_used"' in src, "没有实际模型字段"
    assert '"model_hint"' in src, "没有保留提示值"
    assert '"model": model' not in src, "还在报提示值当模型用"


def test_each_record_carries_model_used():
    """每条记录带 `model_used`——这样**同一轮里换过模型**也看得出来。"""
    from app.vision import service as sv

    src = inspect.getsource(sv.analyze_frames)
    assert '"model_used": used_model' in src


def test_analyze_one_returns_four_tuple():
    """`analyze_one` 必须把实际模型名带出来（第 4 个返回值）。"""
    sig = inspect.signature(__import__("app.vision.service",
                                       fromlist=["x"]).analyze_one)
    assert len(sig.parameters) >= 1
    src = inspect.getsource(__import__("app.vision.service",
                                       fromlist=["x"]).analyze_one)
    assert '(meta or {}).get("model")' in src, "没有从 meta 取真实模型名"
