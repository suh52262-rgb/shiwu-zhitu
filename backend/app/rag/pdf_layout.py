"""PDF 版面几何：文字行、图注识别、插图区域定位。

**这个模块不调用任何模型。** 纯几何 + 文字层，因此：
  · 可离线、可重复、零成本
  · 结果稳定（模型每次给的框都不一样）

## 为什么插图定位要走这一路

40 页抽样显示，两家视觉模型给出的插图 bbox **两两一致率平均只有 IoU 0.563**
（28/44 对低于 0.6）——bbox 定位是整个导入流程里最不可靠的一环。
而图注的位置是**从文字层精确拿到**的。

## 为什么用「包含率」而不是 IoU 判定好坏

第 47 页（人工核对过正解）上的对照：

| 方法 | 包含率（不切图） | IoU（贴合度） |
|---|---|---|
| **图注锚点（本模块）** | **100%** | 0.72 |
| glm-4.6v | 98% | 0.94 |
| deepseek-chat | 93% | 0.51 |
| qwen3-vl-30b-a3b | **91%** | **0.91** |

**qwen 的 IoU 最高但包含率最低——它切掉了 9% 的图。**
裁图是给学生看的，**切掉图比留白严重得多**，所以按包含率选。
"""
from __future__ import annotations

import re
from typing import Any

#: 图注：中文教材的 "图 1-17 xxx"。
#: 要求出现在**行首**才是图注——正文里引用图的句子
#: （"（见图1－17中的黑线轨迹…"）不在行首，不算。
CAPTION_RE = re.compile(r"图\s*\d+\s*[-–—－]\s*\d+[^\n，。；]{0,40}")

#: 图注结尾出现这些标点，说明是从正文里截到的引用（如 "图6-70)"），不是图注。
CAPTION_TAIL_JUNK = re.compile(r"[)\]）】,，、。;；]")

#: 判定「正文横穿」的阈值。实测区分度很大：绕图排版的窄栏文字是 0%，
#: 横穿行是 70%~100%，中间很空，阈值不敏感。
CROSS_FRAC = 0.35

#: 比较时把图注 x 范围朝正文栏方向放宽的比例（占页宽）。
#: 用于处理"图排在正文栏右侧、正文行够不到图注"的版式（见 locate_figure_by_caption）。
LEFT_SLACK_RATIO = 0.09


def text_lines(page: Any) -> list[dict]:
    """行级文字（含 bbox）。

    **必须用行级而不是块级。** 实测 OCR 会把公式行与相邻正文合并成
    70pt 以上的块（如 `'y = c tanα(1.15)2v,°cos'α这是抛物线方程…'`），
    块级 bbox 拿来做几何判断会失准。
    """
    out: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                out.append({"text": text, "bbox": [round(v, 1) for v in line["bbox"]]})
    return out


def caption_is_real(text: str) -> bool:
    """图注是不是真的独立图注，而不是正文里的一句引用。

    `图6-70)` 这种带右括号的，是正文 "（见图6-70)" 被行首正则截到的。
    它推出来的"图"只有 5pt 高——不排掉就会产出垃圾裁图。
    """
    return not CAPTION_TAIL_JUNK.search(text)


def find_captions(page: Any) -> list[dict]:
    """捞图注（行级 + 行首 + 排除正文引用）。"""
    captions: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            match = CAPTION_RE.match(text)          # match = 必须从行首开始
            if match and caption_is_real(match.group(0).strip()):
                captions.append(
                    {"text": match.group(0).strip(),
                     "bbox": [round(v, 1) for v in line["bbox"]]}
                )
    return captions


def locate_figure_by_caption(
    page: Any,
    caption_bbox: list[float],
    *,
    cross_frac: float = CROSS_FRAC,
    left_slack_ratio: float = LEFT_SLACK_RATIO,
    x_pad_frac: float = 0.35,
    min_h: float = 20.0,
    max_h_ratio: float = 0.65,
) -> list[float] | None:
    """用**图注几何**推出插图区域。推不出返回 None（此时才交给模型）。

    中文教材的图注在图**下方**。正文绕图排版时，图旁边的文字行 x 范围
    够不到图注的 x 范围；一旦正文重新横穿过来，说明图已结束。

        y_top    = 第一条「横穿图注 x 范围」的文字行的 y1
        y_bottom = 图注的 y0
        x 范围   = 图注 x 范围向两侧外扩（图比图注宽）

    第 47 页实测：图注 `[564,550,814,575]`，第一条横穿行 y1=398，
    人工核对的正解图 y=`[398,533]` —— **上边界对到 1pt 以内**。

    ## 判定"横穿"时要**朝正文栏方向放宽图注的 x 范围**

    全量 492 页跑出 24 页（4.9%）失败，其中一类是这样的：
    图排在正文栏**右边**的空白区，图注在图下方（如第 64 页图注
    `x=[463,745]`，而所有正文行的 x 最大只到 435）。
    这种版式下**正文行永远够不到图注的 x 范围**，重叠恒为 0，
    怎么调阈值都失败。

    所以比较时把图注 x 范围**向左放宽 `left_slack_ratio × 页宽`**——
    让"贴着图左侧的正文行"也算作横穿，从而正确找到图的上边界。
    实测这一改让第 64 / 80 / 125 页都成功了。

    ## 一类**不该**修的失败

    图注上方几 pt 就是正文（如第 94 页，最近横穿行距图注 5pt）——
    说明**图不在图注正上方**（可能在图左边或右侧）。这种情况推不出是
    **正确行为**，应当回落到模型给的框，而不是硬凑一个错框。

    ## 已知弱点

    图特别大时（第 404 页图8-20），横穿行离图很远，会把一大片正文
    一起框进来。包含率仍是 100%（不切图），但不够干净。
    这是"宁可多框、不可切掉"的取舍结果。
    """
    cx0, cy0, cx1, _cy1 = caption_bbox
    cap_w = max(1.0, cx1 - cx0)

    def scan(band_x0: float) -> float | None:
        for line in sorted(text_lines(page), key=lambda x: -x["bbox"][3]):
            lx0, _ly0, lx1, ly1 = line["bbox"]
            if ly1 > cy0 - 1:                  # 只看图注**上方**的行
                continue
            overlap = max(0.0, min(lx1, cx1) - max(lx0, band_x0))
            if overlap / cap_w >= cross_frac:
                return ly1
        return None

    def acceptable(top: float | None) -> bool:
        return top is not None and min_h <= cy0 - top <= page.rect.height * max_h_ratio

    # **两遍扫描，先严后宽。**
    # 一上来就放宽会把"贴着图左侧的行"也算成横穿，扫描过早停下、
    # 高度太小而被拒——实测放宽到 0.09 反而从 82.4% 掉到 78.7%。
    # 所以只在严格带失败时才放宽，用来救"图排在正文栏右侧"的版式。
    y_top = scan(cx0)
    if not acceptable(y_top):
        relaxed = scan(cx0 - page.rect.width * left_slack_ratio)
        if acceptable(relaxed):
            y_top = relaxed

    if not acceptable(y_top):
        return None
    height = cy0 - y_top
    if not (min_h <= height <= page.rect.height * max_h_ratio):
        return None

    pad = cap_w * x_pad_frac
    return [
        round(max(0.0, cx0 - pad), 1),
        round(y_top, 1),
        round(min(page.rect.width, cx1 + pad), 1),
        round(cy0, 1),
    ]


def locate_figures(page: Any) -> list[dict]:
    """把一页上所有图注都转成插图区域。推不出的条目会带 `needs_model=True`。"""
    out = []
    for cap in find_captions(page):
        box = locate_figure_by_caption(page, cap["bbox"])
        out.append({
            "label": cap["text"],
            "caption_bbox": cap["bbox"],
            "bbox": box,
            "source": "caption_anchor" if box else "unresolved",
            "needs_model": box is None,
        })
    return out
