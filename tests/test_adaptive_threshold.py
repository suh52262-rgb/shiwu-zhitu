"""自适应变化阈值的测试。

## 为什么要它（实测依据）
固定阈值 `0.04` 在部分素材上**落在本底噪声里**：
```
素材01（深色画板）  本底 P10 = 0.0014   → 0.04 远高于本底，能筛
素材02（平板笔记）  本底 P10 = 0.0632   → **0.04 低于本底，等于不筛**
素材03（平板笔记）  本底 P10 = 0.0523   → 同上
```
02/03 是 1840×2944 而码率只有 3.0 Mbps，**每像素码率低约 40%、压缩伪影更重**。

## ⚠️ 但收益**未经证实**（如实记录）
实测 02：接上自适应后，探测帧里 **3274/3417（96%）被正确判为静止**，
**可整帧数仍是 143**——与接上之前一样。

**说明这段素材上约束帧数的是 `min_gap_sec`，不是阈值。**

所以这个改动**让 `is_change` 真正起作用而不是装饰**，
但"减少停顿期间的无效帧"这个**预期收益在本素材上没体现出来**。
需要"停顿多、`min_gap_sec` 小"的素材才能验证。
"""
import pytest

from app.vision.service import (
    ADAPT_CALIBRATE,
    ADAPT_FACTOR,
    ADAPT_MAX_FACTOR,
    adaptive_change_threshold,
)


def _calib(p10, n=60, spread=0.05):
    """造一个 P10 约为 `p10` 的校准样本。"""
    return [p10 + spread * i / n for i in range(n)]


# ---------------------------------------------------------------- 核心行为

def test_low_noise_video_keeps_base_threshold():
    """**本底低的素材不能被改动**（素材01）。

    它的 P10=0.0014，`0.0014×1.5=0.0021` 远小于 0.04，
    所以阈值应当保持 0.04 —— 自适应只该"抬高"，不该"降低"。
    """
    assert adaptive_change_threshold(0.04, 0, _calib(0.0014)) == pytest.approx(0.04)


def test_high_noise_video_raises_threshold_above_floor():
    """**本底高的素材要被抬到本底之上**（素材02/03）。

    这是本函数存在的理由：0.04 落在 0.063 的本底里等于不筛。
    """
    got = adaptive_change_threshold(0.04, 0, _calib(0.0632))
    assert got > 0.0632, f"阈值 {got} 仍在噪声本底之下，等于没修"
    assert got == pytest.approx(0.0632 * ADAPT_FACTOR, rel=0.35)


def test_calibration_period_returns_base():
    """校准期没满就先按基准走——不能拿一两个样本当本底。"""
    assert adaptive_change_threshold(0.04, 0, [0.5] * 5) == 0.04
    assert adaptive_change_threshold(0.04, 0, []) == 0.04


def test_max_factor_caps_the_adaptation():
    """**必须有上限。**

    一段"从头到尾都在写"的视频没有静止段可估本底，分位数会估得很高，
    阈值骑上去**把真内容也筛掉**。
    """
    got = adaptive_change_threshold(0.04, 0, _calib(0.5))
    assert got == pytest.approx(0.04 * ADAPT_MAX_FACTOR), "上限没生效"
    assert got < 0.5, "阈值不该高过本底本身"


def test_zero_base_means_disabled():
    """`change_threshold=0` = 关闭筛选，要尊重调用方，不能被自适应改成非零。"""
    assert adaptive_change_threshold(0.0, 0, _calib(0.06)) == 0.0


def test_calibration_uses_only_the_prefix():
    """**只用开头那一段**估本底。

    否则后面持续书写的高分会把本底估高——那是"骑上去筛掉真内容"的路径。
    """
    calib = [0.001] * ADAPT_CALIBRATE + [0.9] * 100
    got = adaptive_change_threshold(0.04, 0, calib)
    assert got == pytest.approx(0.04), "用了校准期之外的数据，阈值被抬高了"


def test_constants_are_sane():
    assert ADAPT_CALIBRATE >= 10
    assert 1.0 < ADAPT_FACTOR < 3.0
    assert ADAPT_MAX_FACTOR >= 2.0


# ---------------------------------------------------------------- 接线

def test_loop_uses_adaptive_threshold():
    """**防"写了没接"**——判定循环必须用生效阈值，不是配置值。"""
    import inspect

    from app.vision import service as sv

    src = inspect.getsource(sv._extract_keyframes)
    assert "adaptive_change_threshold(" in src
    assert "is_change = score >= eff_change" in src, \
        "还在用固定阈值判 is_change"
    assert src.count("eff_change") >= 3, "eff_change 没被传到各处"


def test_calib_is_initialized():
    """校准列表必须初始化——漏了会 `NameError`（实测踩过）。"""
    import inspect

    from app.vision import service as sv

    src = inspect.getsource(sv._extract_keyframes)
    assert "calib: list = []" in src


def test_effective_threshold_is_recorded():
    """生效阈值要写进 `change_scores.csv`。

    不然事后看那个文件，会以为用的是配置里的 `change_threshold`。
    """
    import inspect

    from app.vision import service as sv

    src = inspect.getsource(sv._extract_keyframes)
    assert '"thr": round(eff_change, 4)' in src
