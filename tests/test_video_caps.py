"""两个限额各管各的 + 截断必须显式报告。

## 这个 bug 的代价（真实数据实测）
一段 5.6 分钟 / 10148 帧的作答视频，`max_frames=200` 把 **ROI 也算进配额**：
`kept=124` 整帧 + `rois=76` = 200 → 停。
**只分析了前 34 秒（10%）**，而输出里只有 `kept=124`，**看不出被截断**。

修复后：整帧 200、ROI 128、覆盖 54.5 秒（16.1%），
并且**打出警告**说明被截断到哪、为什么。

> 覆盖仍然不足是**另一层问题**（`min_gap_sec=0.25` 让关键帧率高达 3.7/秒）——
> 那需要更多视频才能定参数。**但至少它现在是"响的"，不是静默的。**
"""
import inspect

import pytest

from app.vision.service import VideoAnalysisOptions, _FrameSink


def _sink(**kw):
    o = VideoAnalysisOptions(**kw)
    return _FrameSink.__new__(_FrameSink), o


def test_two_caps_are_separate_options():
    """**两个约束必须分开**：一个管时间线覆盖，一个管成本。"""
    o = VideoAnalysisOptions()
    assert hasattr(o, "max_frames") and hasattr(o, "max_images")
    assert o.max_frames != o.max_images or o.max_frames > 0


def test_max_frames_description_no_longer_says_full_plus_roi():
    """`max_frames` 的说明原先写着"整帧+ROI"——**那个说明就是 bug 本身**。"""
    d = VideoAnalysisOptions.model_fields["max_frames"].description or ""
    assert "整帧+ROI" not in d, "说明还写着旧语义"
    assert "整帧" in d


def test_full_is_limited_by_max_frames_not_max_images():
    """整帧只受 `max_frames` 限。

    若受 `max_images` 限，**成本上限会决定时间线覆盖到哪**——正是这个 bug。
    """
    src = inspect.getsource(_FrameSink.full)
    assert "max_frames" in src
    assert "max_images" not in src, "整帧不该受成本上限限制"


def test_roi_is_limited_by_max_images_not_max_frames():
    """ROI 只受 `max_images` 限——它是成本约束，不该限制时间线。"""
    src = inspect.getsource(_FrameSink.roi)
    assert "max_images" in src
    assert "max_frames" not in src, "ROI 还在吃时间线的配额"


def test_n_full_counts_only_full_frames():
    """`n_full()` 必须排除 ROI——它是"分析了多少个时刻"的唯一口径。"""
    src = inspect.getsource(_FrameSink.n_full)
    assert "roi" in src
    # 真实实例上验证
    o = VideoAnalysisOptions()
    sink = _FrameSink.__new__(_FrameSink)
    sink.options = o
    sink.records = [{"type": "keyframe"}, {"type": "roi"},
                    {"type": "gradient"}, {"type": "roi"}]
    assert sink.n_full() == 2


def test_break_sites_use_n_full():
    """抽取循环里的停止判据必须用 `n_full()`。

    用 `len(sink.records)` 会把 ROI 算进去 → 提前收工。
    """
    src = inspect.getsource(__import__("app.vision.service", fromlist=["x"]))
    assert "len(sink.records) >= options.max_frames" not in src, \
        "还有站点在用 len(sink.records) 判 max_frames"


def test_run_pipeline_reports_coverage():
    """**截断必须显式报告**——静默截断是最难发现的一类问题。"""
    from app.vision import service as sv

    src = inspect.getsource(sv.run_pipeline)
    assert "truncated" in src
    assert "covered_until_sec" in src
    assert "video_total_sec" in src
    assert "covered_ratio" in src
    assert "分析被截断" in src, "没有给用户看的警告文案"


def test_coverage_in_both_summaries():
    """dry-run 与完整模式两条返回路径都要带 coverage。"""
    from app.vision import service as sv

    src = inspect.getsource(sv.run_pipeline)
    assert src.count('"coverage": coverage') == 2, \
        "两个 summary 没有都带上 coverage"
