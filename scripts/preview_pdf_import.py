"""PDF 导入原型：验证"逐页判类型 → 整页 VL → 组装"的最小闭环。

**这是最小验证脚本，先不写数据库**——把真实产物拿出来人工核对，
确认提取质量与成本之后，再决定是否落库。

## 设计依据（都来自实测，见 docs/PDF导入方案.md）

1. **公式整页交给 VL，不做精确裁图。**
   实测：裁图 4/5 覆盖且混进坐标轴被读成 `\int`；整页 5/5 全对。
   根因是 VL 有上下文能推断符号含义，裁图丢失上下文。

2. **插图位置也问 VL，不用"文字覆盖补集"。**
   后者试了两轮都失败：正文两端不齐，短行右侧留白与段落间距在整页内
   连成一片，膨胀 20pt 都堵不住（始终输出一个覆盖 79%~100% 的连通块）。
   而 VL 一次就能给出精准 bbox（实测两张图裁出来都完整、不混正文）。

3. **公式与插图合并到同一次调用**，省一半开销。

4. **图注用行级正则**（要求 `图X-Y` 在行首）。实测图注会与正文合并成块，
   块级正则会把正文里的"（见图1－17中的黑线轨迹…"也当成图注。

用法：
    python scripts/preview_pdf_import.py <pdf> --page 47 [--no-vl] [--out DIR]
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import pymupdf  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# 版面几何（图注识别、插图区域定位）已移到 backend/app/rag/pdf_layout.py
# —— 那部分是纯几何、不调模型，应当在 backend 里有自己的位置，也便于测试。
from app.rag.pdf_layout import (  # noqa: E402
    CAPTION_RE,
    caption_is_real as _caption_is_real,
    find_captions,
    locate_figure_by_caption,
    text_lines,
)


# ---------------------------------------------------------------- 提示词

#: VL 返回的 bbox 是归一化到 0~1000 的坐标
NORM = 1000.0

# 提示词放在 prompts/page_reader/v1.md（项目约定），不重复写一份
def _page_prompt() -> str:
    from app.agent.prompt_registry import get_prompt
    return get_prompt("page_reader")


PAGE_PROMPT = _page_prompt()


# ---------------------------------------------------------------- ① 逐页判类型

def page_kind(page) -> str:
    """判定页面是原生文字、扫描件（有/无文字层）。"""
    images = page.get_images(full=True)
    text = page.get_text("text").strip()
    full_page_image = False
    for info in images[:1]:
        for rect in page.get_image_rects(info[0]):
            cover = (rect.width * rect.height) / (page.rect.width * page.rect.height)
            if cover >= 0.8:
                full_page_image = True
    if full_page_image and len(text) > 200:
        return "scanned_with_ocr"
    if full_page_image:
        return "scanned_no_text"
    return "native_text" if text else "empty"


def extract_text_blocks(page) -> list[dict]:
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = "".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if text:
            blocks.append({"text": text, "bbox": [round(v, 1) for v in block["bbox"]]})
    return blocks


class PageRead(BaseModel):
    """VL 返回的结构。用 pydantic 承接，才能走 lenient_parse 的规范化。"""

    formulas: list[str] = Field(default_factory=list)
    figures: list[dict] = Field(default_factory=list)


def read_page(page, dpi: int = 200) -> tuple[dict, dict]:
    """整页渲染 → VL → 公式 + 插图位置。返回 (解析结果, 用量)。

    模型与候选链在 `app.services.vision` 里统一管理——**不在脚本里写死模型名**
    （已被模型变更打过两次脸：qwen-vl-max 定位不可用、qwen3-vl-flash 下架）。

    **解析必须用 `lenient_parse` 而不是 `json.loads`。** 实测 40 页里
    模型会产出未转义的控制字符（公式里的换行）、被截断的字符串——
    严格解析会把这些判成"失败"，而它们其实能救回来。
    项目在 P20 就做好了 `lenient_parse`，这里没接上是我先前的疏漏：
    40 页里有 1 页严格解析失败但宽松解析成功。
    """
    from app.services import vision
    from app.services.llm.fallback import lenient_parse

    pix = page.get_pixmap(dpi=dpi)
    image = pix.tobytes("png")

    text, meta = vision.call_vision(image, PAGE_PROMPT)
    meta = dict(meta, raw_chars=len(text))

    value, notes = lenient_parse(text, PageRead)
    if value is None:
        return {
            "formulas": [], "figures": [],
            "parse_error": "输出无法解析（严格与宽松都失败）",
            "raw": text,
        }, meta

    data = value.model_dump()
    if notes:
        data["parse_notes"] = [str(n) for n in notes]
    return data, meta


def crop_figures(page, figures: list[dict], out_dir: Path, page_no: int) -> list[dict]:
    """按 VL 给的归一化 bbox 裁插图。"""
    sx = page.rect.width / NORM
    sy = page.rect.height / NORM
    saved = []
    for i, fig in enumerate(figures, 1):
        bbox = fig.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        rect = pymupdf.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        # 太小的多半是误判（图标、装饰线）
        if rect.width < 40 or rect.height < 40:
            continue
        path = out_dir / f"p{page_no}_fig{i}.png"
        page.get_pixmap(clip=rect, dpi=150).save(str(path))
        saved.append({
            "path": path.name,
            "label": str(fig.get("label", "")),
            "bbox_norm": [round(v) for v in bbox],
            "bbox_page": [round(v, 1) for v in rect],
        })
    return saved


# ---------------------------------------------------------------- 主流程

def import_page(pdf: Path, page_index: int, out_dir: Path, use_vl: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(pdf)) as doc:
        page = doc[page_index]
        result = {
            "pdf": pdf.name,
            "page": page_index + 1,
            "page_kind": page_kind(page),
            "size": [round(page.rect.width), round(page.rect.height)],
        }

        text_blocks = extract_text_blocks(page)
        result["text_blocks"] = len(text_blocks)
        result["chars"] = sum(len(b["text"]) for b in text_blocks)
        result["text"] = "\n".join(b["text"] for b in text_blocks)
        result["captions"] = find_captions(page)

        if not use_vl:
            result["formulas"] = []
            result["figures"] = []
            return result

        data, meta = read_page(page)
        result["vl"] = meta
        result["formulas"] = data.get("formulas", [])
        if data.get("parse_error"):
            result["parse_error"] = data["parse_error"]
        result["figures"] = crop_figures(page, data.get("figures", []), out_dir,
                                         page_index + 1)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF 导入原型（最小验证）")
    parser.add_argument("pdf", help="PDF 路径")
    parser.add_argument("--page", type=int, default=1, help="页码（从 1 开始）")
    parser.add_argument("--no-vl", action="store_true", help="跳过 VL（只看正文与图注）")
    parser.add_argument("--out", help="产物目录")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"✗ 找不到 {pdf}")
        return 1
    out_dir = Path(args.out) if args.out else (
        Path(r"C:\Users\admix\AppData\Local\Temp\_import_out") / pdf.stem
    )

    t0 = time.perf_counter()
    result = import_page(pdf, args.page - 1, out_dir, use_vl=not args.no_vl)
    total = time.perf_counter() - t0

    print("=" * 78)
    print(f"{result['pdf']}  第 {result['page']} 页   页面类型={result['page_kind']}")
    print("=" * 78)
    print(f"  正文 {result['text_blocks']} 块 / {result['chars']} 字")
    print(f"  图注 {len(result['captions'])} 条：")
    for c in result["captions"]:
        print(f"      {c['text'][:58]}")
    if not args.no_vl:
        m = result.get("vl", {})
        print(f"  VL：{m.get('model')}，{m.get('elapsed_s')}s，图 {m.get('image_kb')} KB，"
              f"tokens={m.get('prompt_tokens')}+{m.get('completion_tokens')}")
        if result.get("parse_error"):
            print(f"  ⚠ 解析问题：{result['parse_error']}")
        print(f"  公式 {len(result['formulas'])} 条：")
        for i, f in enumerate(result["formulas"][:8], 1):
            one = " ".join(f.split())
            print(f"      [{i}] {one[:92]}")
        print(f"  插图 {len(result['figures'])} 张：")
        for f in result["figures"]:
            print(f"      {f['path']:<16} {f['label']:<10} bbox={f['bbox_page']}")
    print(f"\n  单页总耗时 {total:.1f}s    产物目录 {out_dir}")

    (out_dir / f"page_{result['page']}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
