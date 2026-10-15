"""PDF 导入编排：PDF → 内容块 → PostgreSQL（事实）+ Qdrant（索引）。

## 分工（详见 docs/PDF导入方案.md 第八节）

    PostgreSQL  document_blocks  ← 事实来源（结构化、可重跑、可审计）
    Qdrant      textbook_chunks  ← 索引（向量 + 展示用 payload）
    文件系统     data/documents/{doc_id}/figures/  ← 原图

**为什么落两层**：整本教材导入要花约 1000 万 tokens。只存 Qdrant 的话，
将来换 embedding 模型就得重调一遍 VL。先落 PostgreSQL，
重建索引只是重新 embed 文本，不用再碰模型。

## 每一页怎么处理（各路依据都来自实测，见方案文档）

1. **判页面类型**（原生文字 / 扫描件）
2. **正文**  → 文字层（行级）
3. **公式**  → **整页交给 VL**。实测：裁图 4/5 覆盖且混进坐标轴被读成 `\\int`，
   整页 5/5 全对——VL 有上下文能推断符号含义。
4. **图注**  → 文字层行首正则（不依赖模型）
5. **插图**  → **优先用图注锚点定位**（纯几何、零模型调用、包含率 100%）；
   只有推不出的才回落到模型给的框。实测模型的两两一致率只有 IoU 0.563，
   是全流程最不可靠的一环。

## 已知未完成

- 双栏 PDF 的分栏还没做（实测的教材是单栏）
- 试卷切题（题目切分 → problem.json）还没接
- `figure_description`（VL 生成的图注）还没做——先不做，
  因为它是机器生成内容，进检索要额外处理 `is_generated`
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.rag.formula_text import formula_embed_text
from app.rag.import_guard import verify_figure
from app.rag.pdf_layout import find_captions, locate_figure_by_caption, text_lines

logger = logging.getLogger(__name__)

#: 正文块合并阈值：同一段落的行拼成一个块，避免碎片化
PARAGRAPH_GAP = 8.0

CHAPTER_RE = re.compile(r"^\s*(第\s*[一二三四五六七八九十百\d]+\s*章[^\n]{0,30})")


@dataclass
class PageResult:
    """一页的处理结果（还没落库）。"""

    page: int
    kind: str = "native_text"
    blocks: list[dict] = field(default_factory=list)
    vl_meta: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: **这一页是不是彻底失败了**（没有产出可用块）。
    #:
    #: 必须与 `notes` 分开：`notes` 里还有"图注锚点定位失败已回落模型"
    #: 这类**不影响可用性**的提示。靠 notes 文本猜"是否失败"会让
    #: 调用方在错误的地方做决定（踩过：全量替换把失败页的旧内容删掉了）。
    failed: bool = False


def block_id(doc_id: str, page: int, index: int) -> str:
    """幂等主键：同一份文档同一页同一序号，重导覆盖而不是新增。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"shiwu-doc:{doc_id}:{page}:{index}"))


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


def _paragraphs(page) -> list[dict]:
    """把行级文字合并成段落块（行距小且左边界接近的归为一段）。"""
    lines = text_lines(page)
    out: list[dict] = []
    current: dict | None = None
    current_left: float | None = None
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        if current is None:
            current = {"text": line["text"], "bbox": list(line["bbox"])}
            current_left = x0
            continue
        gap = y0 - current["bbox"][3]
        # Preserve the original first-line column criterion while the enclosing
        # box expands to include unindented continuation lines.
        same_col = abs(x0 - current_left) < 30
        if -1 <= gap <= PARAGRAPH_GAP and same_col:
            current["text"] += line["text"]
            current["bbox"][0] = min(current["bbox"][0], x0)
            current["bbox"][1] = min(current["bbox"][1], y0)
            current["bbox"][2] = max(current["bbox"][2], x1)
            current["bbox"][3] = max(current["bbox"][3], y1)
        else:
            out.append(current)
            current = {"text": line["text"], "bbox": list(line["bbox"])}
            current_left = x0
    if current:
        out.append(current)
    return out


def _page_prompt() -> str:
    """整页提示词。

    放在 `prompts/page_reader/v1.md`（项目约定：提示词不写在 Python 里，
    便于版本化与对 A/B）。**不要把它写回脚本再反向 import**——
    那会让 backend 依赖 scripts，分层就反了（踩过这个坑）。
    """
    from app.agent.prompt_registry import get_prompt

    return get_prompt("page_reader")


def _read_page_with_vl(page, dpi: int = 200) -> tuple[dict, dict]:
    """整页交给 VL：拿公式 + 插图初判。**解析必须用 lenient_parse**——
    模型会产出未转义控制字符与截断字符串，严格 json.loads 会误判成失败。"""
    from pydantic import BaseModel, Field

    from app.services import vision
    from app.services.llm.fallback import lenient_parse

    class PageRead(BaseModel):
        formulas: list[str] = Field(default_factory=list)
        figures: list[dict] = Field(default_factory=list)

    text, meta = vision.call_vision(page.get_pixmap(dpi=dpi).tobytes("png"),
                                    _page_prompt())
    value, notes = lenient_parse(text, PageRead)
    if value is None:
        return {"formulas": [], "figures": [], "parse_error": True}, meta
    data = value.model_dump()
    if notes:
        data["parse_notes"] = [str(n) for n in notes]
    return data, meta


def _extract_formula_number(latex: str) -> str:
    # VL commonly emits LaTeX tags rather than parenthesized printed numbers.
    # Conflicting numbers remain unknown; do not silently pick the first one.
    patterns = (r"\\tag\s*\{\s*(\d+[.．\-–]\d+)\s*\}",
                r"[（(]\s*(\d+[.．\-–]\d+)\s*[）)]")
    numbers = {number.replace('．', '.') for pattern in patterns
               for number in re.findall(pattern, latex or '')}
    return f"({next(iter(numbers))})" if len(numbers) == 1 else ""


#: 从 "图1-17抛体运动的轨迹" / "图 3-28" 里抽出图号 "1-17"
_FIG_NUM_RE = re.compile(r"图\s*(\d+)\s*[-–—－]\s*(\d+)")


def _figure_number(text: str) -> str:
    """抽出图号用于配对。

    **不能拿整串比**:VL 返回的 label 是 `图1-17`，文字层的图注是
    `图1-17抛体运动的轨迹`——按完全相等配对会一条都对不上，
    结果同一张图既进"锚点"又进"模型"两条块（实测重复过一次）。
    """
    m = _FIG_NUM_RE.search(text or "")
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def collect_page(
    page,
    page_no: int,
    *,
    doc_id: str,
    use_vl: bool = True,
    figures_dir: Path | None = None,
) -> PageResult:
    """处理一页，产出内容块（不落库）。"""
    result = PageResult(page=page_no, kind=page_kind(page))
    W, H = page.rect.width, page.rect.height
    index = 0

    def add(**kw) -> None:
        nonlocal index
        # doc_id 必须进块本身：Qdrant 的按文档过滤与按文档删除都靠 payload.doc_id，
        # 之前只放在结果信封里，导致按 doc_id 查不到任何点（踩过）。
        result.blocks.append({"id": block_id(doc_id, page_no, index),
                              "doc_id": doc_id, "page": page_no,
                              "block_index": index, **kw})
        index += 1

    # ---- 章节（用于给公式与图注补上下文）----
    section = ""
    for para in _paragraphs(page):
        m = CHAPTER_RE.match(para["text"])
        if m:
            section = m.group(1).strip()
            break

    # ---- ① 正文 ----
    # 扫描件的文字层来自 OCR，公式基本不可用。**保留原文，但不索引**——
    # 否则 `'s = (vo +v)t =( ++at)'` 这种会被当作"教材原文"返回给学生。
    # 阈值在真实页面上核对过，见 app/rag/formula_filter.py。
    from app.rag.formula_filter import classify_text_line

    is_scanned = result.kind.startswith("scanned")
    for para in _paragraphs(page):
        indexable, reason = classify_text_line(para["text"], is_scanned=is_scanned)
        add(type="text", text=para["text"], embed_text=para["text"],
            bbox=para["bbox"], section=section, source_kind="text_layer",
            indexable=indexable, note=reason)

    # ---- ② 图注（文字层，不依赖模型）----
    captions = find_captions(page)

    # **图注文本不再保留一份可索引的正文块。**
    # `_paragraphs()` 会把图注那一行也抽成正文段落，于是同一条图注
    # 既是 text 块又是 figure_caption 块，两份都进索引
    # （实测真实教材 58.7% 的图注如此）。
    # 这里把重复的那份标为不可索引——**原文仍留在 PG**，只是不进检索。
    caption_texts = {(c["text"] or "").strip() for c in captions if c.get("text")}
    if caption_texts:
        for b in result.blocks:
            if b["type"] == "text" and (b["text"] or "").strip() in caption_texts:
                b["indexable"] = False
                b["note"] = "与图注块内容重复，只保留图注块可检索"
        result.notes.append(f"{len(caption_texts)} 条图注已去重（正文块不再重复索引）")

    # ---- ③ 插图：优先图注锚点 ----
    anchored: dict[str, list[float]] = {}
    for cap in captions:
        box = locate_figure_by_caption(page, cap["bbox"])
        if box:
            anchored[cap["text"]] = box

    # ---- ④ 公式与未定位插图：整页 VL ----
    vl_figures: list[dict] = []
    if use_vl:
        data, meta = _read_page_with_vl(page)
        result.vl_meta = meta
        if data.get("parse_error"):
            # **必须标失败，不能只记警告。**
            # 这页的正文块还在，但**公式与插图全丢了**；整文档替换会拿
            # "有正文没公式"的新内容覆盖"正文公式都全"的旧内容——
            # 静默的数据丢失（审计发现）。
            result.failed = True
            result.notes.append(
                "VL 输出无法解析：本页公式与插图缺失（正文已入库，但这一页不完整）"
            )
        for latex in data.get("formulas", []):
            latex = latex.strip()
            if not latex:
                continue
            number = _extract_formula_number(latex)
            add(type="formula", text=latex,
                embed_text=formula_embed_text(latex, page=page_no,
                                              chapter=section, number=number),
                formula_number=number, section=section, source_kind="vl",
                model=meta.get("model", ""))
        vl_figures = data.get("figures", [])

    # ---- ⑤ 插图块 ----
    figures_dir = figures_dir or Path("figures")
    fig_root = figures_dir.parent     # 存库路径相对**文档根**，校验也用文档根
    used_vl: set[int] = set()

    def _figure_ok(path: str) -> tuple[bool, str]:
        """**落库前校验图片引用**。

        `_save_figure` 在裁切失败时返回空路径，但不校验就写块的话，
        PG 里会留下一个指向不存在文件的图注——检索命中时前端拿到死链，
        而"导入成功"的表面完全看不出来。
        （这个函数此前写了却没接上：全仓只有定义、零调用点。）
        """
        if not path:
            return False, "插图裁切失败（无输出文件）"
        _rel, problem = verify_figure(path, fig_root)
        if problem:
            return False, f"插图引用无效：{problem}"
        return True, ""

    for seq, (cap_text, box) in enumerate(anchored.items(), start=1):
        path = _save_figure(page, box, figures_dir, page_no, seq)
        ok, problem = _figure_ok(path)
        if not ok:
            result.notes.append(f"图注「{cap_text[:20]}」{problem}")
        add(type="figure_caption", text=cap_text,
            embed_text=f"{cap_text} 第{page_no}页 {section}".strip(),
            bbox=box, figure_path=path if ok else "",
            figure_labels=[cap_text], section=section,
            source_kind="caption_anchor", indexable=ok, note=problem)

    # 图注锚点推不出的，回落到模型给的框。**按图号配对，不按整串相等**。
    anchored_nums = {_figure_number(t) for t in anchored}
    unmatched = [f for f in vl_figures
                 if _figure_number(str(f.get("label", ""))) not in anchored_nums]
    for i, fig in enumerate(unmatched):
        bbox_norm = fig.get("bbox")
        if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
            continue
        box = [bbox_norm[0] / 1000 * W, bbox_norm[1] / 1000 * H,
               bbox_norm[2] / 1000 * W, bbox_norm[3] / 1000 * H]
        label = str(fig.get("label", ""))
        path = _save_figure(page, box, figures_dir, page_no, 100 + i)
        ok, problem = _figure_ok(path)
        if not ok:
            result.notes.append(f"模型给的图「{label[:20]}」{problem}")
        add(type="figure_caption", text=label,
            embed_text=f"{label} 第{page_no}页 {section}".strip(),
            bbox=[round(v, 1) for v in box],
            figure_path=path if ok else "",
            figure_labels=[label] if label else [],
            section=section, source_kind="vl", indexable=ok, note=problem)
        used_vl.add(i)

    if captions and not anchored:
        result.notes.append("有图注但锚点定位全部失败，已回落到模型框")
    return result


def _save_figure(page, box: list[float], out_dir: Path, page_no: int, seq: int) -> str:
    """裁出插图并返回相对路径（存库用相对路径，便于整体搬迁）。

    **落盘后立刻校验**：写进 PG 的路径如果指向不存在的文件，检索命中时
    前端会拿到死链，而这在"导入成功"的表面上看不出来。
    文件名以 `p{页码}_{序号}` 开头，便于按文档+页码定位。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"p{page_no:04d}_{seq}.png"
    target = out_dir / name
    try:
        page.get_pixmap(clip=pymupdf.Rect(box), dpi=150).save(str(target))
    except Exception as exc:  # noqa: BLE001 单张图失败不该毁掉整页
        logger.warning("第 %d 页第 %d 张插图裁切失败：%s", page_no, seq, exc)
        return ""
    # 正斜杠：存库的路径要跨平台，且便于整目录搬迁
    return f"{out_dir.name}/{name}"


def content_hash(blocks: list[dict]) -> str:
    """内容指纹：用来判断"这一页的提取结果有没有变"，避免无谓重嵌。"""
    h = hashlib.sha256()
    for b in blocks:
        h.update(b.get("text", "").encode("utf-8", "ignore"))
        h.update(b.get("embed_text", "").encode("utf-8", "ignore"))
    return h.hexdigest()[:32]


def import_pdf(
    pdf_path: str | Path,
    doc_id: str,
    *,
    doc_type: str = "textbook",
    pages: list[int] | None = None,
    use_vl: bool = True,
    out_root: str | Path | None = None,
    progress=None,
) -> dict:
    """把一份 PDF 处理成内容块（**只提取，不落库**）。

    落库交给 `persist_blocks()` / `index_blocks()`，
    这样"提取"与"存储"可以分开重跑——提取贵（花 tokens），存储便宜。
    """
    from app.rag.import_guard import safe_join

    pdf_path = Path(pdf_path)
    # **必须走 safe_join**：doc_id 会被拼进输出目录，
    # 不校验的话 `--doc-id ../../..` 能写到仓库外（`ProblemRepository`
    # 早有 _is_safe_id 处理同类问题，导入这条路当时漏了）。
    root = Path(out_root) if out_root else safe_join(Path("data/documents"), doc_id)
    figures_dir = root / "figures"

    all_blocks: list[dict] = []
    page_results: list[PageResult] = []
    with pymupdf.open(str(pdf_path)) as doc:
        targets = pages or list(range(1, doc.page_count + 1))
        for page_no in targets:
            if not (1 <= page_no <= doc.page_count):
                continue
            try:
                pr = collect_page(doc[page_no - 1], page_no, doc_id=doc_id,
                                  use_vl=use_vl, figures_dir=figures_dir)
            except Exception as exc:  # noqa: BLE001 单页失败不该毁掉整本
                logger.warning("第 %d 页处理失败：%s", page_no, exc)
                pr = PageResult(page=page_no, failed=True,
                                notes=[f"处理失败：{type(exc).__name__}"])
            if not pr.blocks and not pr.failed:
                # 一页什么块都没有 = 这页等于没导入。必须显式标失败，
                # 否则它会静默消失在"导入成功"里。
                pr.failed = True
                pr.notes.append("本页未产出任何内容块")
            page_results.append(pr)
            all_blocks.extend(pr.blocks)
            if progress:
                progress(page_no, len(targets), pr)

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "pdf": pdf_path.name,
        "pages": len(page_results),
        "blocks": all_blocks,
        "page_results": page_results,
        "figures_dir": str(figures_dir),
        "hash": content_hash(all_blocks),
    }
