"""`min_gap_sec` 默认值 与 四边裁剪。

## ① `min_gap_sec: 0.25 → 2.0`（用人工标注的步骤时间线验证）
```
02弹簧（294s，步骤在 29s 与 216s）
  gap=1.0 → 218 帧，覆盖到 209.7s（**仍截断**）
  gap=2.0 → 158 帧，覆盖 100%，**两个步骤都抓到**
03平板车（186s，步骤在 110s）
  gap=2.0 →  90 帧，覆盖 100%，步骤抓到
```
**调大间隔既解决覆盖、又减少帧数——是省，不是费。**

## ② 四边裁剪
工具栏位置随 App 而异：
· 深色画板（`真实例子作答01`）—— **顶部**
· 平板笔记（`02/03`，1840×2944 竖屏）—— **左侧**

只裁顶部会漏掉左侧那条工具栏——而它正是 01 里造成 100% 污染的那类东西。
"""
import pytest

from app.vision.service import VideoAnalysisOptions, _FrameSink


def _sink(**kw):
    s = _FrameSink.__new__(_FrameSink)
    s.options = VideoAnalysisOptions(**kw)
    return s


class _F:
    """最小假帧：只需要 shape。"""

    def __init__(self, h, w):
        self.shape = (h, w, 3)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            ys, xs = key
            h = len(range(*ys.indices(self.shape[0])))
            w = len(range(*xs.indices(self.shape[1])))
            return _F(h, w)
        return _F(len(range(*key.indices(self.shape[0]))), self.shape[1])


# ---------------------------------------------------------------- ① 间隔

def test_min_gap_default_is_two_seconds():
    """**默认必须是 2.0。** 0.25 会让 5 分钟的视频要 1200+ 帧，
    而 `max_frames=200` 只能覆盖 16~27%，且**静默截断**。"""
    assert VideoAnalysisOptions().min_gap_sec == 2.0


def test_min_gap_rationale_in_code():
    """实测依据要留在代码里——否则下一个人会"顺手调回 0.25 更细"。"""
    import inspect

    src = inspect.getsource(VideoAnalysisOptions)
    assert "gap=2.0" in src or "158 帧" in src, "缺少实测依据"
    assert "自适应" in src, "没有写明 change_threshold 的已知待办"


# ---------------------------------------------------------------- ② 裁剪

def test_four_edges_available():
    o = VideoAnalysisOptions()
    for f in ("crop_top", "crop_left", "crop_bottom", "crop_right"):
        assert hasattr(o, f), f"缺少 {f}"


def test_no_crop_is_identity():
    f = _F(100, 200)
    assert _sink()._crop_edges(f) is f


def test_crop_each_edge():
    f = _F(100, 200)
    assert _sink(crop_top=0.10)._crop_edges(f).shape[0] == 90
    assert _sink(crop_bottom=0.10)._crop_edges(f).shape[0] == 90
    assert _sink(crop_left=0.10)._crop_edges(f).shape[1] == 180
    assert _sink(crop_right=0.10)._crop_edges(f).shape[1] == 180


def test_crop_cannot_erase_the_frame():
    """**保护**：两边加起来裁光时要放弃裁剪，而不是产出空图。

    `top=0.5` + `bottom=0.5` 是**合法的字段值**（Pydantic 只限单边 ≤0.5），
    但合起来会把画面裁没——所以护栏必须在代码里。
    """
    f = _F(100, 200)
    assert _sink(crop_top=0.5, crop_bottom=0.5)._crop_edges(f).shape[0] == 100
    assert _sink(crop_left=0.5, crop_right=0.5)._crop_edges(f).shape[1] == 200


def test_full_uses_the_crop_helper():
    """**防"写了没接"**。"""
    import inspect

    src = inspect.getsource(_FrameSink.full)
    assert "_crop_edges" in src, "full() 没有用上四边裁剪"
