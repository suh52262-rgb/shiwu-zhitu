"""视觉模型选型与候选链（跨供应商）。

这里断言的是**行为契约**，不是某个具体模型名——模型会下架，但
"不能写死、要能跨供应商回退、要有记账"这几条不该变。
"""
import sys
import types

import pytest

from app.services import vision


# ---------------------------------------------------------------- 候选链

def test_default_model_is_the_measured_best():
    """默认值必须是实测最好的那个，且不能是下架中的 flash。

    这不是"记住一个模型名"，而是：**默认值必须验证过**。
    flash 曾是性价比最优选但 2026-10-10 下架——所以默认值要钉住，
    改动必须是有意识的（并同步更新这里与文档）。
    """
    assert vision.DEFAULT_VISION_MODEL == "deepseek-flash"
    assert vision.VISION_FALLBACKS[0].model == vision.DEFAULT_VISION_MODEL
    assert "qwen3-vl-flash" not in vision.DEFAULT_VISION_MODEL


def test_flash_is_kept_only_as_a_fallback():
    """下架前它仍可用，留在链尾救急有价值；但不能回到主力位置。"""
    models = [c.model for c in vision.VISION_FALLBACKS]
    assert "qwen3-vl-flash" in models
    assert models.index("qwen3-vl-flash") > 0


def test_chain_spans_multiple_providers():
    """**不锁定单一厂商**——这是"模型会变"的结构性应对。

    单页实测：阿里最好、DeepSeek 公式最快最便宜但定位不行、智谱基本可用。
    三家风格不同，互为退路比任何一家单独都稳。
    """
    providers = {c.provider for c in vision.VISION_FALLBACKS}
    assert {"dashscope", "deepseek", "zhipu"} <= providers, providers


def test_every_candidate_points_at_a_known_provider():
    for candidate in vision.VISION_FALLBACKS:
        assert candidate.provider in vision.PROVIDERS, candidate
        assert vision.PROVIDERS[candidate.provider].key_env, candidate


def test_qwen_is_kept_but_not_as_the_only_source():
    """qwen3 的 bbox 最准（0.942），但**在第 146 页把 4 张插图全漏了**。

    所以它适合做插图精修，不适合当唯一来源——漏检比框松严重得多。
    """
    models = [c.model for c in vision.VISION_FALLBACKS]
    assert vision.DEFAULT_VISION_MODEL != "qwen3-vl-30b-a3b-instruct"
    assert "qwen3-vl-30b-a3b-instruct" in models


# ---------------------------------------------------------------- 覆盖

def test_env_override_wins_and_supports_provider_prefix(monkeypatch):
    monkeypatch.setenv("QWEN3VL_MODEL", "zhipu/glm-4.6v")
    assert vision.vision_candidates() == [vision.Candidate("zhipu", "glm-4.6v")]


def test_env_override_bare_name_defaults_to_dashscope(monkeypatch):
    monkeypatch.setenv("QWEN3VL_MODEL", "some-model")
    assert vision.vision_candidates() == [vision.Candidate("dashscope", "some-model")]


def test_chain_has_no_duplicates(monkeypatch):
    monkeypatch.delenv("QWEN3VL_MODEL", raising=False)
    chain = vision.vision_candidates()
    assert len(chain) == len(set(chain))


def test_available_when_any_provider_has_a_key(monkeypatch):
    """只要**任意一个**候选的 key 配好就算可用——这才是"多供应商"的意义。"""
    for provider in vision.PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "fake")
    assert vision.available() is True


def test_unavailable_when_no_provider_has_a_key(monkeypatch):
    for provider in vision.PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    assert vision.available() is False


# ---------------------------------------------------------------- 失败汇报

def test_missing_key_is_reported_per_candidate(monkeypatch):
    """没配 key 的候选要单独说明缺哪个 key，而不是笼统失败。"""
    for provider in vision.PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    with pytest.raises(vision.VisionUnavailable) as excinfo:
        vision.call_vision(b"\x89PNG", "prompt")
    message = str(excinfo.value)
    assert "QWEN3VL_API_KEY" in message


def test_all_candidates_failing_reports_every_reason(monkeypatch):
    """全部失败时，消息里要能看到**每个**候选的原因，而不是一句"就是不行"。"""
    for provider in vision.PROVIDERS.values():
        monkeypatch.setenv(provider.key_env, "fake")

    class Boom:
        def __init__(self, *a, **kw):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError(f"模型 {kwargs['model']} 不可用")

    fake = types.ModuleType("openai")
    fake.OpenAI = Boom
    monkeypatch.setitem(sys.modules, "openai", fake)

    with pytest.raises(vision.VisionUnavailable) as excinfo:
        vision.call_vision(b"\x89PNG", "prompt")

    message = str(excinfo.value)
    for candidate in vision.VISION_FALLBACKS:
        assert candidate.model in message, f"失败原因里缺少 {candidate}"


def test_empty_reply_is_treated_as_failure(monkeypatch):
    """**"接受图片却不报错"的模型必须当失败处理。**

    deepseek-v4-pro 就是这种：它接受 image_url 不报错，
    只在 reasoning 里嘀咕"显示[Unsupported Image]…可能无法看到图片"。
    若把空回复当成成功，就会静默产出空结果。
    """
    for provider in vision.PROVIDERS.values():
        monkeypatch.setenv(provider.key_env, "fake")

    class Silent:
        def __init__(self, *a, **kw):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    class R:
                        choices = [types.SimpleNamespace(
                            message=types.SimpleNamespace(content=""))]
                        usage = None
                    return R()

    fake = types.ModuleType("openai")
    fake.OpenAI = Silent
    monkeypatch.setitem(sys.modules, "openai", fake)

    with pytest.raises(vision.VisionUnavailable, match="可能不支持图片输入"):
        vision.call_vision(b"\x89PNG", "prompt")


# ---------------------------------------------------------------- 记账

def test_cost_estimate_uses_the_right_price_row():
    """记账要按模型取单价——不同档位差十几倍，取错会严重误导成本判断。"""
    cheap = vision.estimate_cost(1_000_000, 1_000_000, "qwen3-vl-flash")
    mid = vision.estimate_cost(1_000_000, 1_000_000, "qwen3-vl-30b-a3b-instruct")
    pricey = vision.estimate_cost(1_000_000, 1_000_000, "qwen3-vl-plus")

    assert cheap < mid < pricey
    assert mid == pytest.approx(0.108 + 0.43)


def test_cost_estimate_tolerates_provider_prefix():
    assert vision.estimate_cost(1_000_000, 0, "dashscope/qwen3-vl-30b-a3b-instruct") \
        == pytest.approx(0.108)


def test_cost_estimate_returns_zero_for_unknown_model():
    """未知模型返回 0 而不是瞎猜——宁可记账缺失，也不要报一个假数字。"""
    assert vision.estimate_cost(1000, 1000, "no-such-model") == 0.0
