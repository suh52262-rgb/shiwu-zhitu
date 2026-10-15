"""视频关键帧判定与 ROI 定位（**纯 CV，与视觉模型无关**）。

从成员 C 的 `C用户交接/analyze_video_opt.py` 抽出。**只抽纯 CV 部分**——
调模型那一半走项目已有的 `app/services/vision.py`，不重复实现。

## 为什么值得抽出来

作者做了几件有实测依据的事，这些结论**与用哪个模型无关**，
值得作为项目资产留下来：

1. **不用 HSV 的 H 通道做直方图** —— 低饱和度（近灰）像素的色相在数学上
   不稳定，±5 的压缩噪声就能让 H 跨多个 bin，**相同画面之间的
   Bhattacharyya 距离高达 0.8（假阳性）**。改用 RGB 各通道 4 分箱量化 +
   预模糊后：静止帧对 ~0.002，跨场景 ~0.3。

2. **局部分数要双尺度** —— 粗尺度（64×36 池化取最活跃 0.3% 网格）抓成片变化；
   细尺度（128×72 取峰值）抓 1–3px 描边的小字，
   **否则会被粗尺度稀释到 0.04 以下**。

3. **Otsu 阈值要夹逼在 [30, 80]** —— 小字出现时差分像素占比可能只有 0.05%，
   **Otsu 在这种极端不均衡下会失灵**（阈值掉进噪声里或吞掉文字）。

4. **关键帧要和「时间邻域探测帧」比，不是和上一个关键帧比** ——
   后者会漏掉缓慢渐变。

## 四路变化分
```
总分 = 0.35*直方图 + 0.15*phash + 0.25*结构 + 0.25*局部
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# ---------------- 权重（作者的实测组合）----------------
W_HIST = 0.35
W_PHASH = 0.15
W_STRUCT = 0.25
W_LOCAL = 0.25

#: 256px 灰度图的边长；结构距离在这一尺度上算
GRAY_SMALL = 256
#: phash 的 DCT 尺寸（16×16 = 256 bit）
PHASH_SIZE = 16
#: 局部分数：粗网格 / 取最活跃的比例；细网格倍数
LOCAL_GRID = (64, 36)
LOCAL_TOP_RATIO = 0.003
LOCAL_FINE_SCALE = 2

#: NCC 在近乎纯色帧上无定义（std 太小），此时退化为像素差
NCC_MIN_STD = 2.0

#: 自动 ROI 的 Otsu 阈值夹逼区间。见模块 docstring 第 3 条。
OTSU_CLIP = (30.0, 80.0)

#: 变化连通块的最小面积占比。**原版默认 0.0003（0.03%）**——
#: 小字出现的差分像素占比可能只有 0.05%，阈值取大了会把该抓的过滤掉。
#: （我第一版凭印象写成 0.002，合成测试里直接返回 None——**默认值要查，不要猜**。）
MIN_AREA_RATIO = 0.0003


@dataclass
class ChangeScore:
    """一帧与参照帧的变化分（各项可解释，便于调阈值时看分布）。"""

    total: float = 0.0
    histogram: float = 0.0
    phash: float = 0.0
    structure: float = 0.0
    local: float = 0.0


def phash(gray_small: np.ndarray, hash_size: int = PHASH_SIZE) -> np.ndarray:
    """DCT 感知哈希（16×16=256bit），抗压缩噪声。"""
    img = cv2.resize(gray_small, (hash_size * 4, hash_size * 4),
                     interpolation=cv2.INTER_AREA)
    dct = cv2.dct(img.astype(np.float32))
    low = dct[:hash_size, :hash_size]
    return low > np.median(low)


def probe_signatures(frame: np.ndarray, *, max_side: int = 0) -> dict[str, Any]:
    """单帧的全部比对签名：RGB 量化直方图 / phash / 256px 灰度 / 全分辨率灰度。

    ⚠️ **不要改成 HSV 的 H 通道**（见模块 docstring 第 1 条）。
    """
    # Experimental detection-only resize. It never changes the caller's OCR
    # image or ROI source. Small strokes can be attenuated, so default is off.
    if max_side > 0 and max(frame.shape[:2]) > max_side:
        scale = max_side / max(frame.shape[:2])
        frame = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)),
                                   max(1, round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray_full, (GRAY_SMALL, GRAY_SMALL),
                            interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(frame, (5, 5), 0)
    hist = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)   # 归一为分布
    return {"hist": hist, "phash": phash(gray_small),
            "gray": gray_small, "gray_full": gray_full}


def ncc_distance(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    """归一化互相关结构距离：0=完全一致，1=完全无关。对布局/形状变化敏感。"""
    if a_gray.shape != b_gray.shape:
        b_gray = cv2.resize(b_gray, (a_gray.shape[1], a_gray.shape[0]))
    mad = float(np.mean(cv2.absdiff(a_gray, b_gray))) / 255.0
    if float(a_gray.std()) < NCC_MIN_STD or float(b_gray.std()) < NCC_MIN_STD:
        return mad          # 近乎纯色帧，NCC 无定义
    try:
        ncc = float(cv2.matchTemplate(a_gray, b_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
    except Exception:       # noqa: BLE001 尺寸异常等
        return mad
    return float(np.clip((1.0 - ncc) / 2.0, 0.0, 1.0))


def localized_score(gray_a: np.ndarray, gray_b: np.ndarray,
                    grid: tuple[int, int] = LOCAL_GRID,
                    top_ratio: float = LOCAL_TOP_RATIO) -> float:
    """局部变化分（**双尺度池化**）：专抓「小区域突变」。

    全局指标对"小字出现/消失"几乎无感，这个分数是为此设计的。

    实测参考：纯噪声背景 ~0.01；小字出现 ~0.1–0.4；场景切换 ~0.3–1.0。
    """
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    diff = cv2.absdiff(gray_a, gray_b)

    cells = cv2.resize(diff, grid,
                       interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    k = max(4, int(round(cells.size * top_ratio)))
    top_mean = float(np.sort(cells.ravel())[::-1][:k].mean())

    fine = cv2.resize(diff, (grid[0] * LOCAL_FINE_SCALE,
                             grid[1] * LOCAL_FINE_SCALE),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    peak = float(fine.max())

    return float(np.clip(max(top_mean, peak), 0.0, 1.0))


def compare(cur: dict[str, Any], prev: dict[str, Any] | None) -> ChangeScore:
    """与上一签名比较，返回四路分与总分。"""
    if prev is None:
        return ChangeScore()
    h = float(cv2.compareHist(prev["hist"], cur["hist"], cv2.HISTCMP_BHATTACHARYYA))
    p = float(np.mean(cur["phash"] != prev["phash"]))
    s = ncc_distance(cur["gray"], prev["gray"])
    loc = localized_score(cur["gray_full"], prev["gray_full"])
    total = W_HIST * h + W_PHASH * p + W_STRUCT * s + W_LOCAL * loc
    return ChangeScore(float(np.clip(total, 0.0, 1.0)), h, p, s, loc)


def unsharp(frame: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask：提升文字边缘对比度（小字能否被读出来的关键之一）。"""
    if amount <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), 1.6)
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def resize_keep(frame: np.ndarray, target_w: int, max_side: int) -> np.ndarray:
    """按目标宽度等比缩放（**只缩不放**），最长边不超过 max_side。"""
    h, w = frame.shape[:2]
    scale = min(1.0, target_w / w) if w else 1.0
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if max(nw, nh) > max_side:
        s2 = max_side / max(nw, nh)
        nw, nh = max(1, int(nw * s2)), max(1, int(nh * s2))
    if (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return frame


def crop_zoom(frame: np.ndarray, box: tuple[int, int, int, int], zoom: float,
              min_side: int, max_side: int, denoise: bool = False,
              sharpen: float = 0.0) -> np.ndarray | None:
    """裁剪 ROI 并放大：最短边尽量 ≥ min_side，最长边 ≤ max_side。

    **200×20 的字幕条能被放大到 1600×160** —— 这是小字能否被 VL 读出来的
    决定性一步。画面外的框会被夹回边界（不报错）。
    """
    x, y, w, h = box
    H, W = frame.shape[:2]
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    scale = max(1.0, float(zoom), min_side / float(min(h, w)))
    if max(h, w) * scale > max_side:
        scale = max(1.0, max_side / float(max(h, w)))
    if scale > 1.01:
        roi = cv2.resize(roi, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_CUBIC)
    if denoise:
        roi = cv2.bilateralFilter(roi, 5, 40, 40)
    return unsharp(roi, sharpen)


def auto_roi_box(prev_frame: np.ndarray | None, cur_frame: np.ndarray,
                 min_area_ratio: float = MIN_AREA_RATIO
                 ) -> tuple[int, int, int, int] | None:
    """差分 + 阈值 + 形态学，找「变化最集中的最大连通块」→ (x,y,w,h)。

    ⚠️ **Otsu 结果必须夹逼在 `OTSU_CLIP`**（见模块 docstring 第 3 条）。

    与**时间邻域探测帧**比，不是与上一个关键帧比——后者会漏掉缓慢渐变。
    """
    if prev_frame is None:
        return None
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    diff = cv2.GaussianBlur(cv2.absdiff(g1, g2), (5, 5), 0)
    otsu_thr, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu_thr, *OTSU_CLIP))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8),
                            iterations=2)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = g1.shape[:2]
    min_area = min_area_ratio * H * W
    cands = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cands:
        return None
    x, y, w, h = cv2.boundingRect(max(cands, key=cv2.contourArea))
    mx, my = max(6, int(W * 0.01)), max(6, int(H * 0.01))
    x, y = max(0, x - mx), max(0, y - my)
    w, h = min(W - x, w + 2 * mx), min(H - y, h + 2 * my)
    return (x, y, w, h)


# ---------------------------------------------------------------- 判定策略

#: 帧类型
KIND_KEYFRAME = "keyframe"      # 内容变化（与上一探测帧比）
KIND_SCENE = "scene"            # 场景切换
KIND_GRADIENT = "gradient"      # 只与锚帧比才有变化 → 缓慢渐变（数字滚动等）
KIND_STATIC_REP = "static_rep"  # 长时间静止补一张代表帧


@dataclass
class KeyframePolicy:
    """判定阈值。默认值来自作者的实测。"""

    probe_every: int = 2              # 每 N 帧探测一次
    scene_threshold: float = 0.20     # ≥ 此值记为场景切换
    change_threshold: float = 0.04    # ≥ 此值记为内容变化
    min_gap_sec: float = 1.0          # 两张保留帧的最小间隔
    static_every_sec: float = 30.0    # 静止多久补一张代表帧（0=不补）
    max_frames: int = 0               # 最多保留几张（0=不限）


@dataclass
class KeyframeState:
    """跨帧状态。**纯数据**——判定逻辑不碰视频也不碰模型，所以可单测。"""

    prev_sig: dict[str, Any] | None = None
    anchor_sig: dict[str, Any] | None = None
    last_kept_ts: float | None = None
    kept: int = 0
    scores: list[dict] = field(default_factory=list)


def classify(score: ChangeScore, anchor_total: float, ts: float, *,
             policy: KeyframePolicy, state: KeyframeState) -> str | None:
    """决定这一探测帧要不要保留、以什么类型保留。返回帧类型或 None。

    这是从 `_extract_keyframes` 的循环里抽出来的**纯判定**——
    原版把它和 `cap.read()` 缠在一起，无法单测。
    """
    first = state.prev_sig is None
    total = max(score.total, anchor_total)

    is_scene = total >= policy.scene_threshold
    is_change = total >= policy.change_threshold
    gap_ok = (state.last_kept_ts is None
              or (ts - state.last_kept_ts) >= policy.min_gap_sec)
    need_static = (policy.static_every_sec > 0
                   and state.last_kept_ts is not None
                   and (ts - state.last_kept_ts) >= policy.static_every_sec)

    if first:
        return KIND_KEYFRAME                 # 首帧必留
    if is_scene and gap_ok:
        return KIND_SCENE
    if is_change and gap_ok:
        # 只与锚帧比才有变化 → 缓慢渐变，与"新内容出现"区分开
        return KIND_KEYFRAME if score.total >= policy.change_threshold else KIND_GRADIENT
    if need_static and gap_ok:
        return KIND_STATIC_REP
    return None
