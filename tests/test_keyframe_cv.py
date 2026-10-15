"""视频关键帧判定的纯 CV 层。

从成员 C 的 `analyze_video_opt.py` 抽出。**只抽纯 CV**——
调模型那一半走项目已有的 `app/services/vision.py`。

## 这组测试重点保的三件事

1. **局部分数真的能抓小字** —— 全局指标对"小字出现"几乎无感，
   这是整个设计的目的。用合成数据实测复现。
2. **判定逻辑可测** —— 原版把 `classify` 和 `cap.read()` 缠在一个循环里，
   无法单测；抽出来后每条分支都能钉住。
3. **作者实测出的参数依据留在代码里** —— 如"不要用 HSV 的 H 通道"。
   那些结论与用哪个模型无关，是项目资产。
"""
import inspect

import cv2
import numpy as np
import pytest

from app.vision import keyframe as kf


# ---------------------------------------------------------------- 合成帧

def _canvas(text="HELLO", color=(240, 240, 240), size=(720, 1280)):
    f = np.full((size[0], size[1], 3), color, np.uint8)
    cv2.putText(f, text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 6)
    return f


def _add_small_text(frame, text="1234"):
    f = frame.copy()
    cv2.putText(f, text, (900, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    return f


# ---------------------------------------------------------------- 变化分

def test_identical_frames_score_zero():
    f = _canvas()
    s = kf.compare(kf.probe_signatures(f), kf.probe_signatures(f.copy()))
    assert s.total == pytest.approx(0.0, abs=1e-6)


def test_scene_change_scores_high():
    s = kf.compare(kf.probe_signatures(_canvas(color=(30, 30, 30))),
                   kf.probe_signatures(_canvas()))
    assert s.total > 0.5
    assert s.histogram > 0.5, "色调全变时直方图距离应当很大"


def test_local_score_catches_small_text():
    """**这是整个设计的目的**：全局指标对小字几乎无感，局部分数要能抓到。

    实测：小字出现时 直方图 ~0.002、结构 ~0.002，
    而 **局部 ~0.29** —— 高出一个数量级。
    """
    base = _canvas()
    s = kf.compare(kf.probe_signatures(_add_small_text(base)),
                   kf.probe_signatures(base))
    assert s.local > 0.1, f"局部分数没抓到小字：{s.local}"
    assert s.local > s.histogram * 10, "局部应显著大于全局指标"
    assert s.local > s.structure * 10


def test_first_frame_has_no_score():
    """没有参照帧时总分为 0（首帧的保留由 classify 决定，不由分数决定）。"""
    s = kf.compare(kf.probe_signatures(_canvas()), None)
    assert s.total == 0.0


def test_localized_score_range():
    a = np.zeros((720, 1280), np.uint8)
    b = np.zeros((720, 1280), np.uint8)
    assert kf.localized_score(a, b) == pytest.approx(0.0)
    c = np.full((720, 1280), 255, np.uint8)
    assert kf.localized_score(a, c) == pytest.approx(1.0)


# ---------------------------------------------------------------- 结构距离

def test_ncc_distance_pure_color_falls_back_to_mad():
    """近乎纯色帧上 NCC 无定义，应退化为像素差而不是给出垃圾值。"""
    a = np.full((64, 64), 10, np.uint8)
    b = np.full((64, 64), 10, np.uint8)
    assert kf.ncc_distance(a, b) == pytest.approx(0.0)
    c = np.full((64, 64), 200, np.uint8)
    d = kf.ncc_distance(a, c)
    assert 0.0 <= d <= 1.0


def test_ncc_distance_handles_shape_mismatch():
    a = np.zeros((64, 64), np.uint8)
    b = np.zeros((32, 32), np.uint8)
    assert 0.0 <= kf.ncc_distance(a, b) <= 1.0


# ---------------------------------------------------------------- ROI

def test_auto_roi_finds_the_changed_region():
    """**核心能力**：自动定位到变化发生的那一小块。"""
    base = _canvas()
    cur = _add_small_text(base)
    box = kf.auto_roi_box(base, cur)
    assert box is not None, "没找到变化区域"
    x, y, w, h = box
    # 小字画在 (900,660) 附近；框应把它包住
    assert x <= 900 <= x + w
    assert y <= 660 <= y + h
    # 而且不该框住整幅画面
    assert w < 1280 * 0.5 and h < 720 * 0.5


def test_auto_roi_none_for_identical_frames():
    f = _canvas()
    assert kf.auto_roi_box(f, f.copy()) is None


def test_auto_roi_none_without_prev():
    assert kf.auto_roi_box(None, _canvas()) is None


def test_min_area_ratio_default_matches_original():
    """默认值**必须来自原版**（0.0003），不能凭印象写。

    我第一版写成 0.002，合成测试里直接返回 None——
    因为小字出现的差分像素占比可能只有 0.05%。
    """
    assert kf.MIN_AREA_RATIO == pytest.approx(0.0003)
    sig = inspect.signature(kf.auto_roi_box)
    assert sig.parameters["min_area_ratio"].default == kf.MIN_AREA_RATIO


def test_crop_zoom_upscales_to_min_side():
    """**小字能否被读出来的决定性一步**：短边必须放大到 min_side。"""
    base = _canvas()
    cur = _add_small_text(base)
    box = kf.auto_roi_box(base, cur)
    out = kf.crop_zoom(cur, box, zoom=1.0, min_side=160, max_side=2400)
    assert out is not None
    assert min(out.shape[:2]) >= 160


def test_crop_zoom_respects_max_side():
    """`max_side` 是给**放大倍数**封顶，不是给输出尺寸封顶。

    ⚠️ 所以我第一版测试写错了：断言"大 ROI 会被压到 max_side"——
    实际不会，而且**不该会**。ROI 是原生分辨率的裁剪，
    缩小会丢掉我们正想读的小字。
    """
    # 大 ROI + 高倍数 → 放大被 max_side 拦住
    out = kf.crop_zoom(_canvas(), (0, 0, 1280, 720), zoom=10.0,
                       min_side=160, max_side=800)
    assert max(out.shape[:2]) == 1280, "大 ROI 不该被缩小（只放大不缩小）"

    # 小 ROI + 高倍数 → 放大后不超过 max_side
    out2 = kf.crop_zoom(_canvas(), (0, 0, 100, 50), zoom=100.0,
                        min_side=160, max_side=800)
    assert max(out2.shape[:2]) <= 800
    assert max(out2.shape[:2]) > 100, "应当被放大"


def test_crop_zoom_never_shrinks():
    """只放大不缩小——原图已经够大时不该被压小。"""
    out = kf.crop_zoom(_canvas(), (0, 0, 400, 400), zoom=1.0,
                       min_side=10, max_side=4000)
    assert out.shape[0] >= 400


def test_crop_zoom_clamps_out_of_bounds_box():
    """画面外的框要被夹回边界，而不是报错。"""
    out = kf.crop_zoom(_canvas(), (1200, 700, 500, 500), zoom=1.0,
                       min_side=10, max_side=4000)
    assert out is not None


def test_resize_keep_only_shrinks():
    f = _canvas()
    assert kf.resize_keep(f, 640, 4096).shape[1] == 640
    # target 比原来大 → 不放大
    assert kf.resize_keep(f, 4000, 4096).shape[1] == 1280
    # max_side 封顶
    assert max(kf.resize_keep(f, 4000, 300).shape[:2]) <= 300


def test_unsharp_zero_is_identity():
    f = _canvas()
    assert kf.unsharp(f, 0.0) is f


# ---------------------------------------------------------------- 判定状态机

def _policy(**kw):
    p = kf.KeyframePolicy()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_first_frame_is_always_kept():
    st = kf.KeyframeState()
    kind = kf.classify(kf.ChangeScore(), 0.0, 0.0, policy=_policy(), state=st)
    assert kind == kf.KIND_KEYFRAME


def test_scene_change_classified_as_scene():
    st = kf.KeyframeState(prev_sig={"x": 1})
    kind = kf.classify(kf.ChangeScore(total=0.9), 0.0, 10.0,
                       policy=_policy(), state=st)
    assert kind == kf.KIND_SCENE


def test_small_change_classified_as_keyframe():
    st = kf.KeyframeState(prev_sig={"x": 1})
    kind = kf.classify(kf.ChangeScore(total=0.07), 0.0, 10.0,
                       policy=_policy(), state=st)
    assert kind == kf.KIND_KEYFRAME


def test_gradient_detected_only_against_anchor():
    """**缓慢渐变**（数字滚动/进度条）：与上一探测帧比变化很小，
    但与锚帧比已经差很多 → 归为 gradient，与"新内容出现"区分开。"""
    st = kf.KeyframeState(prev_sig={"x": 1})
    kind = kf.classify(kf.ChangeScore(total=0.01), 0.07, 10.0,
                       policy=_policy(), state=st)
    assert kind == kf.KIND_GRADIENT


def test_min_gap_blocks_rapid_duplicates():
    """两张保留帧之间要有最小间隔，否则会刷出一串近乎重复的帧。"""
    st = kf.KeyframeState(prev_sig={"x": 1}, last_kept_ts=10.0)
    kind = kf.classify(kf.ChangeScore(total=0.9), 0.0, 10.2,
                       policy=_policy(min_gap_sec=1.0), state=st)
    assert kind is None


def test_static_representative_frame():
    """长时间静止也要补一张代表帧，否则视频后半段会完全没有画面。"""
    st = kf.KeyframeState(prev_sig={"x": 1}, last_kept_ts=0.0)
    kind = kf.classify(kf.ChangeScore(total=0.0), 0.0, 60.0,
                       policy=_policy(static_every_sec=30.0), state=st)
    assert kind == kf.KIND_STATIC_REP


def test_static_disabled():
    st = kf.KeyframeState(prev_sig={"x": 1}, last_kept_ts=0.0)
    kind = kf.classify(kf.ChangeScore(total=0.0), 0.0, 60.0,
                       policy=_policy(static_every_sec=0.0), state=st)
    assert kind is None


def test_below_threshold_is_skipped():
    st = kf.KeyframeState(prev_sig={"x": 1}, last_kept_ts=0.0)
    kind = kf.classify(kf.ChangeScore(total=0.01), 0.01, 1.0,
                       policy=_policy(static_every_sec=0.0), state=st)
    assert kind is None


# ---------------------------------------------------------------- 逐字输入

def test_per_character_input_exceeds_threshold():
    """**逐字输入必须能被抓到** —— 这是本项目最需要的信号。

    ## 为什么专门测这个

    评审报告对**旧脚本**的阻断级发现是：
    静止去重阈值与"屏幕录制/逐字输入"场景严重不匹配，
    **实测把 17 帧中的 16 帧判为静止丢弃**，会丢掉增量作答过程信号。

    新脚本换了机制（关键帧判定替代固定步长），但**阈值本身够不够低没人验过**。
    如果单字增量低于 `change_threshold`，就是**换一种形式重演同一个问题**。

    实测：每个单字增量总分 0.09–0.20，**是阈值 0.04 的 2–5 倍**；
    局部分数 0.33–0.65，是总分的 3–10 倍。

    ⚠️ 这是**合成数据**（干净渲染文字 + 均匀底色）。
    真实屏幕录制/纸面拍摄有压缩噪声、抖动、光照变化，会抬高本底、
    压缩余量。**真机验证仍然必要**——但至少阈值不至于天生就错。
    """
    def frame_with(text):
        f = np.full((720, 1280, 3), 245, np.uint8)
        cv2.putText(f, text, (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (30, 30, 30), 2)
        return f

    threshold = kf.KeyframePolicy().change_threshold
    line = ""
    prev = frame_with("")
    for ch in "mgh=1/2mv2":
        line += ch
        cur = frame_with(line)
        sc = kf.compare(kf.probe_signatures(cur), kf.probe_signatures(prev))
        assert sc.total >= threshold, (
            f"写入 {line!r} 时总分 {sc.total:.4f} 低于阈值 {threshold}——"
            f"逐字输入会被丢弃，重演旧脚本的阻断级问题"
        )
        # 局部分数应当是主要贡献者（全局指标对小字无感）
        assert sc.local > sc.total * 0.8, "局部分数没起到主导作用"
        prev = cur


def test_measured_rationale_is_kept_in_code():
    """**作者实测出的参数依据要留在代码里。**

    它们与用哪个模型无关，是项目资产——删掉注释就等于丢掉了
    "为什么不用 HSV""为什么 Otsu 要夹逼"这些知识，
    下一个人会重新踩一遍。
    """
    src = inspect.getsource(kf)
    assert "HSV" in src, "缺少「不要用 HSV 的 H 通道」的依据"
    assert "Otsu" in src and "夹逼" in src, "缺少 Otsu 夹逼的理由"
    assert "双尺度" in src, "缺少局部分数为何要双尺度"
    assert "时间邻域" in src, "缺少「与邻域探测帧比而非上一关键帧」的理由"


def test_hist_uses_rgb_three_channel():
    """直方图必须是 RGB 三通道 4 分箱 —— 这是那个 HSV 结论的落地形式。"""
    sig = kf.probe_signatures(_canvas())
    assert sig["hist"].shape == (4, 4, 4)


def test_module_is_model_agnostic():
    """**纯 CV 层不许出现任何调模型的东西。**

    它的价值就在于与模型无关：可单测、可复用、
    不会因为换模型（或 qwen3-vl-flash 下架）而失效。
    """
    src = inspect.getsource(kf)
    for forbidden in ("openai", "dashscope", "api_key", "get_llm", "call_vision"):
        assert forbidden not in src.lower(), f"纯 CV 层里出现了 {forbidden}"
