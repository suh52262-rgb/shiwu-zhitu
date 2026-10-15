"""视频分析服务（接入自成员 C 的 analyze_video_opt.py / video_analysis.py）。

**收敛说明**：纯 CV 部分（变化评分 / ROI 定位 / 放大锐化）统一来自
`app.vision.keyframe`——本模块不再重复实现，只保留服务层：
抽帧调度、VL 调用、结果导出。

原脚本为 CLI 工具（argparse + sys.exit），本模块将其重构为可被 FastAPI
直接调用的服务层，算法逻辑与原版保持一致：

    1) 关键帧判定（keyframe 策略，代替固定步长盲抽）
       密集探测（每 probe_every 帧一次），对每个探测帧计算四路变化分：
         - 颜色直方图距离（RGB 4x4x4 量化 + 预模糊，Bhattacharyya）
         - 感知哈希汉明距离（DCT phash 16x16）
         - 结构距离（256px 灰度图归一化互相关 NCC）
         - 局部变化分（全分辨率差分网格池化，专抓小区域突变 = 小字出现）
       总分 = 0.35*直方图 + 0.15*phash + 0.25*结构 + 0.25*局部
    2) 变化区域自动定位 + 放大（微小文字识别的主力）：差分最大连通块
       裁剪放大后走 OCR 专用提示词，200x20 的字幕条可放大到 1600x160。
    3) 高分辨率（默认 1280px / JPEG q95）+ unsharp 锐化。
    4) 整帧走内容理解提示词、ROI 图走 OCR 提示词，互不干扰。

模型调用走 Qwen-VL（DashScope OpenAI 兼容接口），密钥复用全局配置
`qwen_api_key` / `qwen_base_url`（.env），符合工程约定「密钥禁止硬编码」。
cv2/numpy 为可选依赖：缺失时本模块可被 import（API /health 不受影响），
仅在真正执行抽帧时抛出 VideoAnalysisError。
"""
from __future__ import annotations

import base64
import csv
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.vision import available as vision_available
from app.services.vision import call_vision

logger = logging.getLogger("shiwu.video")

# ---- 可选依赖：延迟报错，保证未安装 cv2 时应用仍能启动 ----
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - 环境差异
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------- 纯 CV 层
#
# **算法不在这里实现**——统一来自 `app.vision.keyframe`（有 29 条测试）。
# 本模块只负责服务层：抽帧调度、VL 调用、结果导出。
# 两份实现的算法已逐常量核对一致，合并后只保留一处，避免各自演化。
#
# ⚠️ 这个导入**必须在上面的 try/except 之外**。
# 放进 try 里的话，`keyframe` 自己 import cv2 失败时抛的 ImportError
# 会被 except 吞掉，把"cv2 没装"变成"模块静默降级"。
from app.vision.keyframe import (  # noqa: E402
    OTSU_CLIP,
    auto_roi_box,
    compare as _compare,
    crop_zoom,
    localized_score as _localized_score,
    ncc_distance as _ncc_distance,
    phash as _phash,
    probe_signatures as _probe_signatures,
    resize_keep,
    unsharp,
)


class VideoAnalysisError(RuntimeError):
    """视频分析失败（文件不可读、参数非法、缺依赖等）。"""


def _require_cv() -> None:
    if cv2 is None or np is None:
        raise VideoAnalysisError(
            "缺少视频处理依赖，请先安装:  pip install opencv-python numpy"
        )


# --------------------------------------------------------------------------- #
# 默认提示词（与原脚本一致）
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 默认提示词
#
# ⚠️ **这是"转录式"提示词，不是"描述式"。** 换掉原来的场景描述式提示词是有
# 实测依据的：原提示词问的是 `scene/objects/text/actions/tags/description`
# （通用画面理解），而作答录屏里**文字最多的地方是顶部工具栏**——
# 于是 17/17 个时刻的 text 全是
# `'颜色 填充 粗细 2 透明度 100% 线条 实线 清空 保存 网格 全屏'`。
#
# **模型没读错，是问错了。** 改成"只转录手写内容、不要描述界面"之后，
# 同一帧读出了 `由题知 μ = -kv²，由牛二知`（deepseek-flash，1.2s/17tok）。
#
# 规则 3（看不清写 `?` 不要猜）对教学场景尤其重要：
# **伪造的作答记录比缺失更糟——它会被下游当成真的。**
# --------------------------------------------------------------------------- #

DEFAULT_PROMPT = """这是一个学生在电子画板/纸面上**手写作答**的画面。

任务：**逐字转录**画面上的全部手写内容（中文、数学公式、图表中的标注文字）。

输出严格的 JSON（不要 Markdown 代码块）：
{
  "text": "转录出的完整内容，公式用普通文本表示（如 μ = -k v²）；没有内容返回空字符串",
  "description": "对这段作答的简短说明（学生在写什么、写到哪一步），30-60 字",
  "has_content": true
}

规则：
1. **只转录手写内容**。不要描述界面、工具栏、按钮、颜色、笔画粗细、背景；
2. 数学公式保持结构（等号、上下标、希腊字母），必要时用 ^ 表示上标；
3. 看不清的字符写成 `?`，**不要猜**——宁可标注不确定，也不要编一个看起来完整的式子；
4. 若画面上确实没有手写内容，`text` 返回空字符串、`has_content` 返回 false；
5. 不要输出任何解释性文字。"""

DEFAULT_OCR_PROMPT = """你是一个高精度 OCR 引擎。这张图可能是从视频里裁剪并放大出来的局部区域，也可能是一整帧画面。

任务：把所有可见文字完整、逐行地识别出来。

输出严格的 JSON（不要 Markdown 代码块）：
{
  "text": "按阅读顺序逐行拼接的完整文字，行间用 \\n 分隔；没有文字返回空字符串",
  "lines": ["逐行数组，与 text 内容一致"],
  "has_text": true,
  "confidence": "high | medium | low",
  "unclear_chars": ["看不清而用 ? 标注的字符或片段"]
}

规则：
1. 逐字抄录，不要改写、不要翻译、不要补全、不要总结；
2. 中文、英文、数字、符号、单位、货币都原样保留；
3. 极小字号或模糊字符：给出最佳判断，把握不大时把该字符写成 `?`；
4. 不要输出任何解释性文字。"""

CSV_FIELDS = [
    ("frame", None),
    ("timestamp_sec", None),
    ("frame_type", None),       # scene / keyframe / gradient / static_rep / roi
    ("change_score", None),
    ("roi_tag", None),
    ("scene", "scene"),
    ("objects", "objects"),
    ("text", "text"),
    ("actions", "actions"),
    ("tags", "tags"),
    ("description", "description"),
    ("error", "error"),
]


# --------------------------------------------------------------------------- #
# 参数模型（对应原脚本 argparse 参数）
# --------------------------------------------------------------------------- #

class VideoAnalysisOptions(BaseModel):
    """视频分析参数。model 为空时使用配置 QWEN_VL_MODEL。"""

    # ---- 抽帧与关键帧策略 ----
    strategy: Literal["keyframe", "fixed"] = "keyframe"
    probe_every: int = Field(2, ge=1, description="每 N 帧探测一次")
    detection_max_side: int = Field(0, ge=0, description="实验性变化检测长边上限，0 保持原分辨率；不改变 OCR/ROI 图片")
    fps: float = Field(2.0, gt=0, description="fixed 模式每秒抽几帧")
    every: Optional[int] = Field(None, ge=1, description="fixed 模式每 N 帧抽 1 帧（覆盖 fps）")
    #: **分析多少个时刻**（整帧数）。
    #:
    #: ROI 是整帧的**衍生物**，不是独立时刻，**不占这个配额**。
    #: 原先它的说明写的是"整帧+ROI"，于是 ROI 吃掉了时间线的配额——
    #: 实测 5.6 分钟的视频 kept=124 + rois=76 = 200 就停，**只跑了前 34 秒**。
    max_frames: int = Field(200, ge=0, description="最多保留多少张整帧，0 不限")
    #: **发多少次模型调用**（整帧 + ROI）。这才是成本/时长的上限。
    #: 与 `max_frames` 分开：一个管"覆盖多少时间线"，一个管"花多少钱"。
    max_images: int = Field(400, ge=0, description="最多产出多少张图（整帧+ROI），0 不限")
    #: **从顶部裁掉的比例**（0~0.5）。
    #:
    #: 画板/录屏类素材顶部常有工具栏，不裁掉的话 VL 会把工具栏文字当成作答
    #: 内容——实测 17/17 个时刻的 text 全是
    #: `'颜色 填充 粗细 2 透明度 100% 线条 实线 清空 保存 网格 全屏'`。
    #:
    #: **默认 0（不裁）**，保持既有行为；对这个画板实测 0.06 正好。
    #: **从顶部裁掉的比例**（0~0.5）。
    #:
    #: 画板/录屏类素材顶部常有工具栏或状态栏，不裁掉的话 VL 会把工具栏文字
    #: 当成作答内容——实测 17/17 个时刻的 text 全是
    #: `'颜色 填充 粗细 2 透明度 100% 线条 实线 清空 保存 网格 全屏'`。
    #:
    #: **默认 0（不裁）**，保持既有行为；深色画板实测 0.06 正好。
    crop_top: float = Field(0.0, ge=0.0, le=0.5,
                            description="从顶部裁掉的比例，用于去掉工具栏/状态栏")
    #: **从左侧裁掉的比例**（0~0.5）。
    #:
    #: ⚠️ **平板笔记类素材的工具栏在左边，`crop_top` 去不掉它。**
    #: 实测 `真实例子作答02/03`（1840×2944 竖屏）：顶部裁 0.06 只去掉了状态栏
    #: （时钟/电量），**左侧那条工具栏仍在画面里**——而它正是 01 里
    #: 造成 100% 污染的那类东西。
    #:
    #: **怎么定这个值：不要看图猜，量列墨迹分布找空白带。** 02 的实测：
    #: ```
    #: x= 36- 72 (2.0-3.9%)  墨迹 7428   ← 工具栏图标（最密）
    #: x= 72-108 (3.9-5.9%)  墨迹  391   ← **几乎空白 = 天然分界**
    #: x=108-144 (5.9-7.8%)  墨迹 1842   ← 内容开始
    #: ```
    #: 空白带中心约 **0.045**（推荐值）；留 0.05 也安全，但离内容边界
    #: 只剩约 16px 余量。
    crop_left: float = Field(0.0, ge=0.0, le=0.5,
                             description="从左侧裁掉的比例，用于去掉侧边工具栏")
    crop_bottom: float = Field(0.0, ge=0.0, le=0.5, description="从底部裁掉的比例")
    crop_right: float = Field(0.0, ge=0.0, le=0.5, description="从右侧裁掉的比例")
    width: int = Field(1280, ge=64, description="整帧送模型宽度")
    scene_threshold: float = Field(0.20, ge=0, le=1)
    change_threshold: float = Field(0.04, ge=0, le=1)
    #: **两张保留图的最小时间间隔（秒）。**
    #:
    #: ## 为什么从 0.25 调到 2.0（有实测依据）
    #: 0.25 意味着最多 **4 帧/秒** —— 一段 5 分钟的视频要 **1200+ 帧**，
    #: 而 `max_frames=200` 只能覆盖 **16~27%**，且**静默截断**。
    #:
    #: 用**人工标注的作答步骤时间线**验证（`真实例子作答02/03`）：
    #: ```
    #: 02（294s，步骤在 29s 与 216s）
    #:   gap=1.0 → 218 帧，覆盖到 209.7s（**仍截断**）
    #:   gap=2.0 → 158 帧，覆盖 100%，**两个步骤都抓到**
    #: 03（186s，步骤在 110s）
    #:   gap=2.0 →  90 帧，覆盖 100%，步骤抓到
    #: ```
    #: **调大间隔既解决覆盖、又减少帧数** —— 是省，不是费。
    #:
    #: ⚠️ **前提**：`change_threshold` 在部分素材上落在本底噪声里
    #: （02 的 P10=0.063、03 的 P10=0.052，都 > 0.04），此时**筛选实际由本参数完成**。
    #: 这意味着停顿期间也会按间隔取帧（内容重复，下游稳定期会滤掉，
    #: 但 VL 调用已经花掉）。**自适应阈值是已知待办。**
    min_gap_sec: float = Field(2.0, ge=0, description="两张保留图最小时间间隔")
    static_every_sec: float = Field(5.0, ge=0, description="静止段补代表帧间隔，0 关闭")
    start: float = Field(0.0, ge=0, description="起始秒")
    end: Optional[float] = Field(None, gt=0, description="结束秒，默认到片尾")
    jpeg_quality: int = Field(95, ge=1, le=100)

    # ---- 微小文字增强 ----
    roi: list[str] = Field(default_factory=list, description="固定 ROI x,y,w,h，可多个")
    auto_roi: bool = True
    roi_zoom: float = Field(3.0, ge=1.0)
    roi_min_side: int = Field(640, ge=64)
    roi_max_side: int = Field(1600, ge=256)
    auto_roi_min_area_ratio: float = Field(0.0003, gt=0, le=1)
    auto_roi_max_area_ratio: float = Field(0.20, gt=0, le=1)
    subtitle_band: float = Field(0.0, ge=0, le=1, description="底部字幕带比例，如 0.16")
    ocr_sharpen: float = Field(0.6, ge=0)
    denoise: bool = False

    # ---- 模型调用 ----
    model: str = Field("", description="视觉模型名，留空用 QWEN_VL_MODEL")
    prompt: str = ""
    ocr_prompt: str = ""
    ocr_only: bool = Field(False, description="所有图只走 OCR 提示词")
    workers: int = Field(4, ge=1, le=16)
    retries: int = Field(4, ge=1)
    timeout: float = Field(120.0, gt=0)
    enable_thinking: bool = False
    thinking_budget: int = Field(4096, ge=1)
    no_json_mode: bool = False
    max_side: int = Field(2000, ge=256, description="送模型图片最长边上限")

    # ---- 运行控制 ----
    dry_run: bool = Field(False, description="只抽帧导图，不调用模型")
    export_change_scores: bool = False
    save_raw: bool = False


def resolve_model(options: VideoAnalysisOptions) -> str:
    """实际使用的视觉模型名。"""
    return options.model.strip() or get_settings().qwen_vl_model











# --------------------------------------------------------------------------- #
# 阶段一：抽帧（关键帧 / 固定步长）+ ROI 细节图
# --------------------------------------------------------------------------- #

def _timed_read(cap, timings):
    """Sequential decoder read, including EOF/limit lookahead reads."""
    started = time.perf_counter()
    try:
        return cap.read()
    finally:
        timings["decode_read_sec"] += time.perf_counter() - started


def _extraction_timings():
    # Mutually exclusive subphases; frame output includes ROI localization,
    # crop/resize/sharpen, JPEG encoding and filesystem writes.
    return {"decode_read_sec": 0.0, "change_detection_sec": 0.0,
            "frame_output_sec": 0.0}


class _FrameSink:
    """统一落盘 + 记录，附带 max-frames 限额与 ROI 生成。"""

    def __init__(self, frames_dir: Path, options: VideoAnalysisOptions, jpeg_params, frame_wh):
        self.frames_dir = frames_dir
        self.options = options
        self.jpeg_params = jpeg_params
        self.W, self.H = frame_wh
        self.records: list[dict] = []
        self.fixed_boxes: list[Tuple[int, int, int, int]] = []
        for spec in options.roi:
            try:
                parts = [int(float(v)) for v in re.split(r"[,\s]+", spec.strip()) if v]
                if len(parts) != 4:
                    raise ValueError
                self.fixed_boxes.append(tuple(parts))  # type: ignore[arg-type]
            except Exception:
                logger.warning("无法解析 ROI '%s'，应为 x,y,w,h", spec)
        if options.subtitle_band > 0:
            bh = max(8, int(self.H * float(options.subtitle_band)))
            self.fixed_boxes.append((0, self.H - bh, self.W, bh))

    def n_full(self) -> int:
        """已保存的整帧数（ROI 不算）。"""
        return sum(1 for r in self.records if r.get("type") != "roi")

    def _crop_edges(self, frame):
        """按 `crop_top` / `crop_left` / `crop_bottom` / `crop_right` 裁掉边框。

        **四边都支持**，因为工具栏的位置随 App 而异：
        · 深色画板（`真实例子作答01`）—— 工具栏在**顶部**
        · 平板笔记（`真实例子作答02/03`，竖屏）—— 工具栏在**左侧**

        每边都做了"裁完不能小于 16 像素"的保护，避免参数写错把画面裁没。
        用 `getattr` 取参数，这样旧调用方（没有这些字段）不受影响。
        """
        o = self.options
        h, w = frame.shape[:2]
        top = int(h * float(getattr(o, "crop_top", 0.0) or 0.0))
        bottom = int(h * float(getattr(o, "crop_bottom", 0.0) or 0.0))
        left = int(w * float(getattr(o, "crop_left", 0.0) or 0.0))
        right = int(w * float(getattr(o, "crop_right", 0.0) or 0.0))
        if top + bottom >= h - 16:
            top = bottom = 0
        if left + right >= w - 16:
            left = right = 0
        if not (top or bottom or left or right):
            return frame
        return frame[top:h - bottom if bottom else h,
                     left:w - right if right else w]

    def full(self, frame, ts: float, raw_idx: int, ftype: str, score: float,
             detail: str = "") -> bool:
        """保存一张整帧。**只受 `max_frames` 限**，受成本上限影响就会本末倒置。"""
        if self.options.max_frames and self.n_full() >= self.options.max_frames:
            return False
        # **先裁掉工具栏/状态栏再缩放。**
        # 不裁的话 VL 会把工具栏文字当成作答内容——深色画板那次
        # 17/17 个时刻的 text 全是工具栏（见 DEFAULT_PROMPT 上方的说明）。
        #
        # **四边都要能裁**：深色画板的工具栏在顶部，而平板笔记类素材
        # （真实例子作答02/03，1840×2944 竖屏）**工具栏在左侧**——
        # 只裁顶部会漏掉它。
        frame = self._crop_edges(frame)
        send = unsharp(resize_keep(frame, self.options.width, self.options.max_side),
                       self.options.ocr_sharpen)
        ok, buf = cv2.imencode(".jpg", send, self.jpeg_params)
        if not ok:
            return False
        jpg = buf.tobytes()
        prefix = {"scene": "sc", "static_rep": "st"}.get(ftype, "kf")
        fname = f"{prefix}_{raw_idx:06d}_t{ts:08.3f}s.jpg"
        (self.frames_dir / fname).write_bytes(jpg)
        self.records.append({
            "index": raw_idx, "ts": round(ts, 3), "file": fname,
            "bytes": jpg, "type": ftype, "score": round(score, 4),
            "detail": detail,
        })
        return True

    def roi(self, frame, ts: float, raw_idx: int, tag: str,
            box: Tuple[int, int, int, int], parent_score: float) -> bool:
        """保存一张 ROI。**只受 `max_images` 限**——它是成本约束。

        拿不到配额时静默跳过（整帧已存下，信息不丢）。
        """
        if self.options.max_images and len(self.records) >= self.options.max_images:
            return False
        roi = crop_zoom(frame, box, self.options.roi_zoom, self.options.roi_min_side,
                        self.options.roi_max_side, self.options.denoise,
                        self.options.ocr_sharpen)
        if roi is None or roi.size == 0:
            return False
        ok, buf = cv2.imencode(".jpg", roi, self.jpeg_params)
        if not ok:
            return False
        jpg = buf.tobytes()
        fname = f"roi_{raw_idx:06d}_t{ts:08.3f}s_{tag}_{box[2]}x{box[3]}.jpg"
        (self.frames_dir / fname).write_bytes(jpg)
        self.records.append({
            "index": raw_idx, "ts": round(ts, 3), "file": fname,
            "bytes": jpg, "type": "roi", "score": round(parent_score, 4),
            "roi_tag": tag, "roi_box": list(box),
        })
        return True

    def auto_rois(self, frame, ref_frame, ts: float, raw_idx: int,
                  parent_score: float) -> int:
        """生成该关键帧的全部 ROI：固定框 + 自动变化区域。返回生成数量。"""
        made = 0
        for i, b in enumerate(self.fixed_boxes):
            made += int(self.roi(frame, ts, raw_idx, f"roi{i}", b, parent_score))
        if self.options.auto_roi and ref_frame is not None:
            box = auto_roi_box(ref_frame, frame, self.options.auto_roi_min_area_ratio)
            if box:
                area_ratio = box[2] * box[3] / float(self.W * self.H)
                if area_ratio <= self.options.auto_roi_max_area_ratio:
                    made += int(self.roi(frame, ts, raw_idx, "auto", box, parent_score))
        return made


#: 自适应阈值的**校准探测数**。用开头这么多个探测帧估计本底噪声。
#:
#: 取 30：`probe_every=2` 时约等于视频开头 2 秒（30fps）——
#: 短到不会错过有价值的内容，长到足够估出分位数。
ADAPT_CALIBRATE = 30

#: 本底估计值要**乘多少**才算"真有变化"。
#: 1.5 是在噪声之上留 50% 余量——太小会被噪声穿透，太大会筛掉真实的小改动。
ADAPT_FACTOR = 1.5

#: 自适应阈值的**上限倍数**：最多抬到基准阈值的这么多倍。
#:
#: **必须有这个上限。** 否则一段"从头到尾都在写"的视频（没有静止段可估本底）
#: 会把分位数估得很高，阈值骑上去**把真内容也筛掉**。
ADAPT_MAX_FACTOR = 8.0


def adaptive_change_threshold(base: float, score: float, calib: list,
                              *, need: int = ADAPT_CALIBRATE,
                              factor: float = ADAPT_FACTOR,
                              max_factor: float = ADAPT_MAX_FACTOR) -> float:
    """返回**本次探测**该用的变化阈值。

    ## 为什么需要它（实测依据）
    固定阈值 `0.04` 在部分素材上**落在本底噪声里**：
    ```
    素材01（深色画板）  本底 P10 = 0.0014   → 0.04 远高于本底，能筛
    素材02（平板笔记）  本底 P10 = 0.0632   → **0.04 低于本底，等于不筛**
    素材03（平板笔记）  本底 P10 = 0.0523   → 同上
    ```
    02/03 是 1840×2944 而码率只有 3.0 Mbps，**每像素码率低约 40%，
    压缩伪影更重**——本底因此抬高了 40 倍。

    ## 做法
    用开头 `need` 个探测帧估本底（取 P10），阈值抬到 `本底 × factor`，
    但**不超过 `base × max_factor`**（见 `ADAPT_MAX_FACTOR` 的理由）。

    实测效果（按各素材的 P10 推算）：
    ```
    01: max(0.04, 0.0014×1.5) = 0.040  ← 不变（本底本来就低）
    02: max(0.04, 0.0632×1.5) = 0.095  ← 抬到本底之上
    03: max(0.04, 0.0523×1.5) = 0.078
    ```
    **既不改动本底低的素材，又能修好本底高的素材。**
    """
    if base <= 0:
        return 0.0                      # 0 = 关闭筛选，尊重调用方
    if len(calib) < need or not calib:
        return base                     # 校准期没满，先用基准
    s = sorted(calib[:need])
    p10 = s[max(0, int(len(s) * 0.10) - 1)]
    return min(max(base, p10 * factor), base * max_factor)


class _VideoClock:
    """Use sequential decoder presentation time; never seek by average FPS."""
    def __init__(self, cap, fps):
        self.cap, self.fps, self.previous = cap, fps, 0.0
        self.fallbacks = 0
        self.pts_frames = 0

    def timestamp(self, index):
        try:
            value = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            valid = (math.isfinite(value) and value >= self.previous
                     and (index == 0 or value > 0))
        except (AttributeError, TypeError, ValueError):
            valid = False
        if valid:
            self.pts_frames += 1
        else:
            value = max(self.previous, index / self.fps)
            self.fallbacks += 1
        self.previous = value
        return value

    def annotate(self, stats):
        stats["timestamp_source"] = ("decoded_pts" if not self.fallbacks else
                                     "mixed_pts_and_frame_index" if self.pts_frames else "frame_index_fallback")
        stats["timestamp_reliable"] = self.fallbacks == 0
        stats["timestamp_fallback_frames"] = self.fallbacks


def _extract_keyframes(cap, src_fps: float, options: VideoAnalysisOptions,
                       sink: _FrameSink, deadline: float) -> dict:
    probe_every = max(1, options.probe_every)
    raw_idx = 0
    prev_sig: Optional[dict] = None
    prev_probe_frame = None
    anchor_sig: Optional[dict] = None
    last_kept_ts: Optional[float] = None
    #: 自适应阈值的校准样本（开头若干个探测帧的变化分）。
    #: 见 `adaptive_change_threshold`——固定阈值在高噪声素材上会失效。
    calib: list = []

    stats: dict = {"probes": 0, "kept": 0, "scene": 0, "keyframe": 0,
                   "gradient": 0, "static_rep": 0, "static_skipped": 0,
                   "rois": 0, "scores": []}

    stats["stop_reason"] = "eof"
    stats["scanned_until_sec"] = None
    timings = stats["timings"] = _extraction_timings()
    clock = _VideoClock(cap, src_fps)
    while True:
        ok, frame = _timed_read(cap, timings)
        if not ok:
            break
        ts = clock.timestamp(raw_idx)
        if ts > deadline:
            stats["stop_reason"] = "range_end"
            break
        stats["scanned_until_sec"] = round(ts, 3)
        if ts < options.start or raw_idx % probe_every != 0:
            raw_idx += 1
            continue

        change_started = time.perf_counter()
        sig = (_probe_signatures(frame, max_side=options.detection_max_side)
               if options.detection_max_side else _probe_signatures(frame))
        # ⚠️ `_compare` 现在返回 `ChangeScore` 数据类（来自 app.vision.keyframe），
        # **不是原来的 5 元组**。收敛时漏改了这里，实测第一次跑真实视频就崩：
        #   TypeError: cannot unpack non-iterable ChangeScore object
        _cs = _compare(sig, prev_sig)
        t_prev, loc_ = _cs.total, _cs.local
        # Immediately after retaining a frame the references are identical.
        # Reuse the exact score rather than repeating full-image comparison.
        t_anchor = (t_prev if anchor_sig is prev_sig else
                    _compare(sig, anchor_sig).total if anchor_sig is not None else 0.0)
        score = max(t_prev, t_anchor)
        stats["probes"] += 1

        # ---- 自适应变化阈值（见 `adaptive_change_threshold`）----
        # 固定阈值在部分素材上会落在本底噪声里（实测 02 的 P10=0.063、
        # 03 的 P10=0.052，都 > 0.04），于是"有变化"永远成立、筛选失效。
        # 用一个短的校准期估计本底，之后把它抬到本底之上。
        eff_change = adaptive_change_threshold(
            options.change_threshold, score, calib,
            need=ADAPT_CALIBRATE, factor=ADAPT_FACTOR,
            max_factor=ADAPT_MAX_FACTOR)
        calib.append(score)

        first = prev_sig is None
        is_scene = score >= options.scene_threshold
        is_change = score >= eff_change
        gap_ok = last_kept_ts is None or (ts - last_kept_ts) >= options.min_gap_sec
        need_static = (options.static_every_sec > 0 and last_kept_ts is not None
                       and (ts - last_kept_ts) >= options.static_every_sec)

        ftype: Optional[str] = None
        if first:
            ftype = "keyframe"                      # 首帧必留
        elif is_scene and gap_ok:
            ftype = "scene"
        elif is_change and gap_ok:
            ftype = "keyframe" if t_prev >= eff_change else "gradient"
        elif need_static and gap_ok:
            ftype = "static_rep"

        timings["change_detection_sec"] += time.perf_counter() - change_started
        kept = False
        if ftype:
            output_started = time.perf_counter()
            kept = sink.full(frame, ts, raw_idx, ftype, score,
                             detail=f"prev={t_prev:.3f} anchor={t_anchor:.3f} loc={loc_:.3f}")
            if kept:
                stats["kept"] += 1
                stats[ftype] += 1
                last_kept_ts = ts
                anchor_sig = sig                     # 新静止段锚帧
                stats["rois"] += sink.auto_rois(frame, prev_probe_frame, ts, raw_idx, score)
                timings["frame_output_sec"] += time.perf_counter() - output_started
                # **只有整帧数决定是否停止分析时间线**。
                # 用 len(sink.records) 会把 ROI 算进去，导致提前收工。
                if options.max_frames and sink.n_full() >= options.max_frames:
                    stats["scores"].append({"ts": round(ts, 3), "score": round(score, 4), "keep": 1})
                    more, _ = _timed_read(cap, timings)
                    stats["stop_reason"] = ("max_frames" if more and clock.timestamp(raw_idx + 1) <= deadline
                                            else "range_end" if more else "eof")
                    break
            else:
                timings["frame_output_sec"] += time.perf_counter() - output_started
        else:
            stats["static_skipped"] += 1

        stats["scores"].append({"ts": round(ts, 3), "score": round(score, 4),
                                "keep": int(kept), "prev": round(t_prev, 4),
                                "anchor": round(t_anchor, 4), "loc": round(loc_, 4),
                                # **生效阈值**——自适应之后它不等于配置里的
                                # `change_threshold`。不记下来的话，事后看
                                # `change_scores.csv` 会以为用的是配置值。
                                "thr": round(eff_change, 4)})
        prev_sig = sig
        prev_probe_frame = frame
        raw_idx += 1

    clock.annotate(stats)
    return stats


def _extract_fixed(cap, src_fps: float, options: VideoAnalysisOptions,
                   sink: _FrameSink, deadline: float) -> dict:
    if options.every and options.every > 0:
        step = options.every
    else:
        step = max(1, int(round(src_fps / max(options.fps, 1e-6))))

    raw_idx = 0
    prev_raw = None
    stats: dict = {"probes": 0, "kept": 0, "rois": 0, "scores": []}
    stats["stop_reason"] = "eof"
    stats["scanned_until_sec"] = None
    timings = stats["timings"] = _extraction_timings()
    clock = _VideoClock(cap, src_fps)
    last_sample_ts = None
    while True:
        ok, frame = _timed_read(cap, timings)
        if not ok:
            break
        ts = clock.timestamp(raw_idx)
        if ts > deadline:
            stats["stop_reason"] = "range_end"
            break
        stats["scanned_until_sec"] = round(ts, 3)
        sample_due = (raw_idx % step == 0 if options.every else
                      last_sample_ts is None or ts - last_sample_ts >= 1 / options.fps)
        if ts >= options.start and sample_due:
            output_started = time.perf_counter()
            kept = sink.full(frame, ts, raw_idx, "keyframe", 0.0)
            if kept:
                stats["kept"] += 1
                last_sample_ts = ts
                stats["rois"] += sink.auto_rois(frame, prev_raw, ts, raw_idx, 0.0)
                timings["frame_output_sec"] += time.perf_counter() - output_started
                # **只有整帧数决定是否停止分析时间线**。
                # 用 len(sink.records) 会把 ROI 算进去，导致提前收工。
                if options.max_frames and sink.n_full() >= options.max_frames:
                    more, _ = _timed_read(cap, timings)
                    stats["stop_reason"] = ("max_frames" if more and clock.timestamp(raw_idx + 1) <= deadline
                                            else "range_end" if more else "eof")
                    break
            else:
                timings["frame_output_sec"] += time.perf_counter() - output_started
        prev_raw = frame
        raw_idx += 1
    clock.annotate(stats)
    return stats


def extract_frames(video_path: Path, frames_dir: Path, options: VideoAnalysisOptions,
                   log: Optional[Callable[[str], None]] = None) -> Tuple[list[dict], dict]:
    """抽帧主入口：返回 (帧记录列表, 统计信息)。帧记录内含 JPEG 字节。"""
    extraction_started = time.perf_counter()
    _require_cv()
    log = log or (lambda msg: None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoAnalysisError(f"无法打开视频：{video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / src_fps if total_frames else 0.0
    deadline = options.end if options.end is not None else duration + 1e9
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or options.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    log(f"[抽帧] 源视频 {video_path.name} | 分辨率 {width}x{height} | "
        f"帧率 {src_fps:.2f}fps | 总帧数 {total_frames} | 时长 {duration:.2f}s")
    if options.strategy == "keyframe":
        log(f"[抽帧] 策略 = 关键帧 | 探测 每{probe_every_disp(options.probe_every)}帧 | "
            f"场景阈值 {options.scene_threshold} | 变化阈值 {options.change_threshold} | "
            f"宽度 {options.width}px | q={options.jpeg_quality}")
    else:
        log(f"[抽帧] 策略 = 固定步长 | fps={options.fps} every={options.every} | "
            f"宽度 {options.width}px | q={options.jpeg_quality}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(options.jpeg_quality)]
    sink = _FrameSink(frames_dir, options, jpeg_params, (width, height))

    if options.strategy == "fixed":
        stats = _extract_fixed(cap, src_fps, options, sink, deadline)
    else:
        stats = _extract_keyframes(cap, src_fps, options, sink, deadline)
    cap.release()
    timings = stats["timings"]
    timings["extraction_total_sec"] = time.perf_counter() - extraction_started
    timings["extraction_other_sec"] = max(0.0, timings["extraction_total_sec"] - sum(
        timings[k] for k in ("decode_read_sec", "change_detection_sec", "frame_output_sec")))

    if not sink.records:
        return [], stats

    n_roi = sum(1 for r in sink.records if r["type"] == "roi")
    note = (f"[抽帧] 完成：整帧 {len(sink.records) - n_roi} 张 + ROI 细节图 {n_roi} 张"
            f" -> {frames_dir}")
    if stats.get("probes"):
        note += (f" | 探测 {stats['probes']} 次 | 场景切换 {stats.get('scene', 0)} | "
                 f"关键帧 {stats.get('keyframe', 0)} | 渐变救回 {stats.get('gradient', 0)} | "
                 f"静止跳过 {stats.get('static_skipped', 0)} | 静止补帧 {stats.get('static_rep', 0)}")
    log(note)
    return sink.records, stats


def probe_every_disp(n: int) -> str:
    return str(n)


def _write_change_scores(path: Path, scores: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp_sec", "score", "kept", "score_prev", "score_anchor", "score_local"])
        for s in scores:
            w.writerow([s.get("ts"), s.get("score"), s.get("keep", ""),
                        s.get("prev", ""), s.get("anchor", ""), s.get("loc", "")])


# --------------------------------------------------------------------------- #
# 阶段二：调用 Qwen-VL 做识别
# --------------------------------------------------------------------------- #

def build_vl_client(options: VideoAnalysisOptions):
    """返回 `(None, model_hint)`。

    **不再自己建客户端。** 视觉调用统一走 `app.services.vision.call_vision`，
    它按 `VISION_FALLBACKS` 依次尝试候选链、按供应商注入正确的思考控制、
    统一记账。

    ## 为什么必须改
    原实现硬性要求 `QWEN_API_KEY`。实测环境里那个 key **是空的**，
    而项目主力是 `deepseek-flash`（`DEEPSEEK_API_KEY` 已配）——
    **于是整条视频 VL 链路直接抛错，根本跑不起来。**

    ## ⚠️ 第二个返回值**只回显显式覆盖**，不是 `resolve_model()`
    这一点很容易搞错，而且我搞错过一次：`resolve_model(options)` 会**套上默认值**
    （`qwen_vl_model`），把它当 `model=` 传给 `call_vision` 的后果是
    **只试那一个候选，`VISION_FALLBACKS` 整条链被绕过**。

    实测：`models_used` 报的是 `qwen3-vl-plus`（配置默认值），
    而不是候选链首选的 `deepseek-flash`——**路由到 `call_vision` 的主要理由
    （拿回退链）根本没生效**。

    所以这里只回显 `options.model`（用户显式指定才会有值）；
    留空则交给 `call_vision` 走完整候选链。
    """
    if not vision_available():
        raise VideoAnalysisError(
            "没有可用的视觉模型：请至少配置 DEEPSEEK_API_KEY / QWEN_API_KEY / "
            "ZHIPU_API_KEY 之一"
        )
    return None, options.model.strip()


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"result": data}
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, dict) else {"result": data}
        except Exception:
            pass
    return None


def analyze_one(client, jpg_bytes: bytes, prompt: str, options: VideoAnalysisOptions,
                model: str, thinking: Optional[bool] = None):
    """单图调用模型，返回 (原始文本, 解析后的 JSON, 错误信息)。"""
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    use_thinking = options.enable_thinking if thinking is None else thinking
    extra_body: dict = {"enable_thinking": bool(use_thinking)}
    if use_thinking:
        extra_body["thinking_budget"] = options.thinking_budget

    kwargs: dict = {"model": model, "messages": messages, "extra_body": extra_body}
    use_json_mode = not options.no_json_mode
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if not use_thinking:
        kwargs["temperature"] = 0.1

    # ---- 交给项目统一的视觉服务 ----
    #
    # 它自带五级回退链，所以这里**不需要自己写重试**（原来那段退避逻辑
    # 是为单一供应商写的，现在由候选链承担）。
    #
    # `extra_body` **故意不传**：`call_vision` 会按供应商注入正确的思考控制
    # （deepseek 默认开思考，实测 22s·6151tok vs 2.7s·291tok）。
    # C 原来的 `{"enable_thinking": False}` 是 Qwen 专有格式，对 deepseek 无效。
    try:
        content, meta = call_vision(
            jpg_bytes, prompt,
            model=model or None,
            timeout=options.timeout,
            temperature=0.1 if not use_thinking else 1.0,
            mime="image/jpeg",      # **帧是 JPEG q95**，不能标成 PNG
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}", model

    content = (content or "").strip()
    if not content:
        return None, None, "视觉服务返回空内容", model
    # **返回实际用的模型名**（`meta` 里的），而不是传进来的提示值。
    # 第 4 个返回值是为了让调用方能如实报告——`call_vision` 走候选链，
    # 实际可能是另一个供应商的模型。
    return (content, _extract_json(content), None,
            (meta or {}).get("model") or model)


def analyze_frames(records: list[dict], main_prompt: str, ocr_prompt: str,
                   options: VideoAnalysisOptions, out_dir: Path, client,
                   model: str,
                   log: Optional[Callable[[str], None]] = None,
                   progress_cb: Optional[Callable[..., None]] = None) -> list[dict]:
    """并发调用视觉模型识别全部帧，返回结果记录列表。

    client / model 由调用方通过 build_vl_client() 构建后传入，
    便于上层复用同一客户端并统一模型名。
    """
    log = log or (lambda msg: None)
    progress_cb = progress_cb or (lambda *a, **k: None)

    results: list[Optional[dict]] = [None] * len(records)
    lock = threading.Lock()
    counter = {"done": 0, "fail": 0}
    total = len(records)

    def work(pos: int, rec: dict) -> None:
        is_roi = rec.get("type") == "roi"
        use_ocr = is_roi or options.ocr_only
        prompt = ocr_prompt if use_ocr else main_prompt

        raw, parsed, err, used_model = analyze_one(
            client, rec["bytes"], prompt, options, model)

        item: dict = {
            "frame": rec["file"],
            "index": rec["index"],
            "timestamp_sec": rec["ts"],
            "frame_type": rec.get("type", ""),
            "change_score": rec.get("score", 0.0),
            "roi_tag": rec.get("roi_tag", ""),
            "roi_box": rec.get("roi_box"),
            "analyze_mode": "ocr" if use_ocr else "content",
            # **实际用的模型**（候选链可能换），不是 `resolve_model()` 的提示值。
            # 记在每条记录上，这样连"同一轮里换过模型"都看得出来。
            "model_used": used_model,
        }
        if parsed:
            if use_ocr:
                item["text"] = parsed.get("text", "")
                item["ocr_lines"] = parsed.get("lines", [])
                item["ocr_confidence"] = parsed.get("confidence", "")
                item["scene"], item["objects"] = "", []
                item["actions"], item["tags"], item["description"] = [], [], ""
            else:
                for key in ("scene", "objects", "text", "actions", "tags", "description"):
                    item[key] = parsed.get(key, "" if key in ("scene", "text", "description") else [])
            for k, v in parsed.items():
                item.setdefault(k, v)
        else:
            for k, v in (("scene", ""), ("objects", []), ("text", ""),
                         ("actions", []), ("tags", []), ("description", "")):
                item[k] = v
            if err:
                item["error"] = err

        if options.save_raw:
            item["raw_response"] = raw

        results[pos] = item
        with lock:
            counter["done"] += 1
            if err:
                counter["fail"] += 1
            n = counter["done"]
        progress_cb("analyzing", f"识别进度 {n}/{total}",
                    analyze_done=n, analyze_total=total)
        preview = str(item.get("text") or item.get("scene") or "")[:40]
        log(f"  [{'OK ' if not err else 'ERR'}] {n}/{total}  {rec['file']}  -> {preview}")
        if err:
            log(f"        错误：{str(err)[:160]}")

    n_roi = sum(1 for r in records if r.get("type") == "roi")
    log(f"[分析] 模型 {model} | 并发 {options.workers} | 共 {total} 张图"
        + (f"（整帧 {total - n_roi} + ROI {n_roi}）" if n_roi else ""))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
        futures = [pool.submit(work, i, rec) for i, rec in enumerate(records)]
        for _ in as_completed(futures):
            pass
    cost = time.perf_counter() - t0

    ok_cnt = sum(1 for r in results if r and not r.get("error"))
    log(f"[分析] 完成：成功 {ok_cnt} / 失败 {total - ok_cnt} | 耗时 {cost:.1f}s "
        f"| 平均 {cost / max(total, 1):.2f}s/图")

    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
# 阶段三：结果导出
# --------------------------------------------------------------------------- #

def _join(v: Any) -> str:
    if isinstance(v, list):
        return " / ".join(str(x) for x in v)
    if v is None:
        return ""
    return str(v).replace("\n", " ").strip()


def export_results(video_path: Path, records: list[dict], all_frames: list[dict],
                   out_dir: Path, stats: Optional[dict] = None, model: str = "",
                   options: Optional[VideoAnalysisOptions] = None,
                   log: Optional[Callable[[str], None]] = None) -> None:
    """导出 results.json / results.csv / change_scores.csv / report.md。"""
    log = log or (lambda msg: None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # **报实际用的模型，不是 `model` 参数里的提示值。**
    # `call_vision` 走候选链，实际可能是别的供应商的模型——
    # 而从每条记录的 `model_used` 汇总是唯一可靠的口径。
    #
    # ⚠️ 这里曾长期写 `"model": model`（提示值），于是**留存下来的证据
    # 指向了错误的模型**：`run_pipeline` 的返回值我修过，但 `results.json`
    # 才是事后被翻出来看的东西。**一份说错话的证据比没有证据更糟。**
    models_used = sorted({r.get("model_used") for r in records
                          if r.get("model_used")})

    payload = {
        "video": str(video_path),
        "video_name": video_path.name,
        "models_used": models_used,
        # 提示值另存，便于对照"想让谁干"与"实际谁干的"
        "model_hint": model,
        "strategy": getattr(options, "strategy", ""),
        "params": {
            "probe_every": getattr(options, "probe_every", None),
            "detection_max_side": getattr(options, "detection_max_side", 0),
            "scene_threshold": getattr(options, "scene_threshold", None),
            "change_threshold": getattr(options, "change_threshold", None),
            "width": getattr(options, "width", None),
            "jpeg_quality": getattr(options, "jpeg_quality", None),
            "roi": getattr(options, "roi", None),
            "auto_roi": getattr(options, "auto_roi", False),
            "roi_zoom": getattr(options, "roi_zoom", None),
            "roi_min_side": getattr(options, "roi_min_side", None),
            "subtitle_band": getattr(options, "subtitle_band", 0.0),
        },
        "full_frames": len([r for r in all_frames if r.get("type") != "roi"]),
        "roi_frames": len([r for r in all_frames if r.get("type") == "roi"]),
        "analyzed": len(records),
        "keyframe_stats": {k: v for k, v in (stats or {}).items() if k != "scores"},
        "frames": records,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([c[0] for c in CSV_FIELDS])
        for r in records:
            row = []
            for col, key in CSV_FIELDS:
                if key is None:
                    row.append(r.get(col, ""))
                else:
                    row.append(_join(r.get(key, "")))
            writer.writerow(row)

    if stats and stats.get("scores") and getattr(options, "export_change_scores", False):
        _write_change_scores(out_dir / "change_scores.csv", stats["scores"])

    # ---- report.md ----
    lines: list[str] = ["# 视频关键帧 + 文字提取报告\n"]
    lines.append(f"- 视频文件：`{video_path.name}`")
    lines.append(f"- 抽取策略：`{getattr(options, 'strategy', '')}`")
    # **实际用的模型**——报告是给人看的，不能用提示值糊弄。
    lines.append(f"- 实际使用的模型：`{', '.join(models_used) or '(未记录)'}`"
                 + (f"（配置/提示为 `{model}`）" if model and model not in models_used
                    else ""))
    if stats and stats.get("probes"):
        lines.append(f"- 探测 {stats.get('probes', 0)} 次 | 场景切换 {stats.get('scene', 0)} | "
                     f"关键帧 {stats.get('keyframe', 0)} | 渐变 {stats.get('gradient', 0)} | "
                     f"静止补帧 {stats.get('static_rep', 0)}")
    lines.append(f"- ROI 细节图：{len([r for r in all_frames if r.get('type') == 'roi'])} 张")
    lines.append(f"- 成功分析：{sum(1 for r in records if not r.get('error'))} 张\n")

    by_ts: dict[float, list[dict]] = {}
    for r in records:
        by_ts.setdefault(float(r.get("timestamp_sec", 0)), []).append(r)

    all_text, all_tags = [], []
    ocr_blocks: list[str] = []
    for ts in sorted(by_ts):
        group = by_ts[ts]
        texts = []
        for r in sorted(group, key=lambda x: 0 if x.get("frame_type") == "roi" else 1):
            t = _join(r.get("text")).strip()
            if not t:
                continue
            if r.get("frame_type") == "roi":
                texts.append(f"[ROI {r.get('roi_tag', '')} @{r.get('roi_box')}] {t}")
            else:
                texts.append(f"[整帧] {t}")
        for r in group:
            tt = _join(r.get("text"))
            if tt:
                all_text.append(tt)
            for tag in (r.get("tags") or []):
                all_tags.append(str(tag))
        if texts:
            ocr_blocks.append(f"### {ts:.2f}s\n" + "\n".join(texts))

    if all_text:
        lines.append("## 画面文字汇总（OCR，含 ROI 增强）\n")
        lines.append("```")
        lines.extend(all_text)
        lines.append("```\n")
    if ocr_blocks:
        lines.append("## 文字时间轴（按关键帧聚合）\n")
        lines.extend(ocr_blocks)
        lines.append("")
    if all_tags:
        top = Counter(all_tags).most_common(20)
        lines.append("## 高频标签 TOP20\n")
        lines.append("  ".join(f"`{k}`×{v}" for k, v in top))
        lines.append("")

    lines.append("## 逐图时间轴\n")
    lines.append("| 时间 | 类型 | 变化分 | OCR 文字 | 场景 | 描述 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in records:
        ts = r.get("timestamp_sec", 0)
        lines.append("| {}s | {} | {} | {} | {} | {} |".format(
            f"{float(ts):.2f}",
            r.get("frame_type", ""),
            r.get("change_score", ""),
            _join(r.get("text"))[:80].replace("|", "／"),
            _join(r.get("scene"))[:40].replace("|", "／"),
            _join(r.get("description"))[:80].replace("|", "／"),
        ))
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"[导出] results.json / results.csv / report.md  ->  {out_dir}")


# --------------------------------------------------------------------------- #
# 主流程（供 API 后台任务调用）
# --------------------------------------------------------------------------- #

def run_pipeline(video_path: Path, out_dir: Path, options: VideoAnalysisOptions,
                 log: Optional[Callable[[str], None]] = None,
                 progress_cb: Optional[Callable[..., None]] = None) -> dict:
    """完整流水线：抽帧 →（dry-run 结束）→ VL 识别 → 导出。返回结果摘要。

    timings top-level phases are disjoint wall-clock durations. Extraction's
    nested subphases are a breakdown, not additional time to add to the total.
    Vision includes concurrent work/retries; it is not the sum of request times.
    """
    pipeline_started = time.perf_counter()
    timings = {"extraction_sec": 0.0, "client_setup_sec": 0.0,
               "vision_sec": 0.0, "export_sec": 0.0}

    def finish_timings():
        total = time.perf_counter() - pipeline_started
        timings["other_sec"] = max(0.0, total - sum(timings[k] for k in (
            "extraction_sec", "client_setup_sec", "vision_sec", "export_sec")))
        timings["total_sec"] = total
        timings["extraction"] = dict(stats.get("timings", {}))
        return timings
    log = log or (lambda msg: None)
    progress_cb = progress_cb or (lambda *a, **k: None)

    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoAnalysisError(f"视频文件不存在：{video_path}")

    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_prompt = options.prompt.strip() or DEFAULT_PROMPT
    ocr_prompt = options.ocr_prompt.strip() or DEFAULT_OCR_PROMPT

    progress_cb("extracting", f"开始抽帧：{video_path.name}")
    stage_started = time.perf_counter()
    frames, stats = extract_frames(video_path, frames_dir, options, log=log)
    timings["extraction_sec"] = time.perf_counter() - stage_started
    if not frames:
        raise VideoAnalysisError("没有抽到任何帧，请检查 start/end/probe_every/fps 参数。")

    n_full = len([r for r in frames if r.get("type") != "roi"])
    n_roi = len(frames) - n_full

    # ---- **截断要显式报告** ----
    # 静默截断是最难发现的一类问题：实测一段 5.6 分钟的视频只处理了前 34 秒
    # （10%），而输出里只有 `kept=124`，看不出被截断。
    # 项目一贯纪律是"不吞掉失败"——静默截断正是那种失败。
    _covered = max((float(r.get("ts") or 0.0) for r in frames), default=0.0)
    _total = 0.0
    try:
        _cap = cv2.VideoCapture(str(video_path))
        if _cap.isOpened():
            _fps = _cap.get(cv2.CAP_PROP_FPS) or 0.0
            _cnt = _cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            _total = round(_cnt / _fps, 3) if _fps else 0.0
        _cap.release()
    except Exception:  # noqa: BLE001 拿不到时长不影响主流程
        pass
    _truncated = stats.get("stop_reason") == "max_frames"
    total_source = "frame_count_divided_by_average_fps_estimate"
    if stats.get("timestamp_reliable") and stats.get("stop_reason") == "eof":
        # The final presentation time is a lower bound, not the final frame's end.
        _total = float(stats.get("scanned_until_sec") or 0)
        total_source = "last_decoded_pts_lower_bound"
    coverage = {
        "truncated": _truncated,
        "timestamp_source": stats.get("timestamp_source", "unknown"),
        "timestamp_reliable": stats.get("timestamp_reliable", False),
        "video_total_source": total_source,
        "extraction_stop_reason": stats.get("stop_reason", "unknown"),
        "requested_start_sec": options.start,
        "requested_end_sec": options.end,
        "scanned_until_sec": stats.get("scanned_until_sec"),
        "covered_until_sec": round(_covered, 3),
        "video_total_sec": _total,
        "covered_ratio": (round(_covered / _total, 4) if _total else None),
        "truncate_reason": (
            f"达到 max_frames={options.max_frames}（整帧）；"
            f"只分析了前 {_covered:.1f}/{_total:.1f} 秒"
            if _truncated else ""),
    }
    if _truncated:
        log(f"[警告] **分析被截断**：{coverage['truncate_reason']}。"
            f"提高 max_frames 或缩小范围（--start/--end）以覆盖更长的视频。")

    if options.dry_run:
        stage_started = time.perf_counter()
        if options.export_change_scores and stats.get("scores"):
            _write_change_scores(out_dir / "change_scores.csv", stats["scores"])
        timings["export_sec"] = time.perf_counter() - stage_started
        log(f"[结束] dry-run 模式，仅抽帧，未调用模型。图片位置：{frames_dir}")
        summary = {
            "video_name": video_path.name,
            "out_dir": str(out_dir),
            "frames_dir": str(frames_dir),
            "full_frames": n_full,
            "roi_frames": n_roi,
            "dry_run": True,
            "timings": finish_timings(),
            "keyframe_stats": {k: v for k, v in stats.items() if k != "scores"},
            "coverage": coverage,
        }
        progress_cb("done", f"抽帧完成（dry-run）：整帧 {n_full} + ROI {n_roi}")
        return summary

    progress_cb("analyzing", f"调用视觉模型识别 {len(frames)} 张图",
                analyze_done=0, analyze_total=len(frames))
    # 提前构建客户端：缺 API Key 等配置问题在抽帧前就暴露
    stage_started = time.perf_counter()
    client, model = build_vl_client(options)
    timings["client_setup_sec"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    records = analyze_frames(frames, main_prompt, ocr_prompt, options, out_dir,
                             client, model, log=log, progress_cb=progress_cb)
    timings["vision_sec"] = time.perf_counter() - stage_started

    progress_cb("exporting", "导出结果文件")
    stage_started = time.perf_counter()
    export_results(video_path, records, frames, out_dir, stats=stats,
                   model=model, options=options, log=log)
    timings["export_sec"] = time.perf_counter() - stage_started

    failed_frames = [{k: r.get(k) for k in ("frame", "index", "timestamp_sec", "error")}
                     for r in records if r.get("error")]
    coverage["failed_frames"] = failed_frames
    coverage["failed_frame_count"] = len(failed_frames)
    ok_cnt = sum(1 for r in records if not r.get("error"))
    summary = {
        "video_name": video_path.name,
        "out_dir": str(out_dir),
        "frames_dir": str(frames_dir),
        "results_json": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
        "report_md": str(out_dir / "report.md"),
        "full_frames": n_full,
        "roi_frames": n_roi,
        "analyzed": len(records),
        "ok": ok_cnt,
        "failed": len(records) - ok_cnt,
        # **实际用的模型**——从每条记录的 `model_used` 汇总。
        # 候选链可能换供应商，所以这里**不能**直接写 `model`（那是提示值）。
        # 保留 `model_hint` 另存提示值，便于对照"想让谁干"与"实际谁干的"。
        "models_used": sorted({r.get("model_used") for r in records
                               if r.get("model_used")}),
        "model_hint": model,
        "dry_run": False,
        "timings": finish_timings(),
        "keyframe_stats": {k: v for k, v in stats.items() if k != "scores"},
        "coverage": coverage,
    }
    progress_cb("done", f"分析完成：成功 {ok_cnt}/{len(records)}")
    return summary
