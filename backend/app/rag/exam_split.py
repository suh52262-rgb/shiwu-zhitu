"""试卷切题：一份试卷 PDF → 一道道题（题干 / 解答 / 配图）。

## 为什么题号识别要用「连续递增」而不是只看模式

`1.` 在中文试卷里既可能是**题号**，也可能是解答内部的编号，
甚至可能是「（1）」这类小问被 OCR 读丢了括号。单看模式必然误切。

所以采用**序列约束**：一个候选起点只有在其编号等于"期望的下一个题号"
（或从 1 重新开始）时才被接受。误报的 `1.` 除非恰好接在上一题后面，
否则不会被当成新题起点。

## 与 P14 的衔接

切出来的每道题写成 `data/problems/{problem_id}/problem.json`（复用既有
`ProblemRepository` 的格式），再交给 P14 自动建图、P15 校验——
于是"导入一份试卷 → 立刻多出 N 道可教学的题"。

## 未做

- 表格题、跨页题的分隔判定还较粗糙
- 图像归属只按"同页 + 就近"处理，跨页图会漏
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.pdf_layout import text_lines

logger = logging.getLogger(__name__)

#: 题号起点。覆盖 `例3.1` / `例题3` / `第3题` / `3.` / `3、` / `3．`
PROBLEM_START = re.compile(
    r"^\s*(?:"
    r"例\s*题?\s*(\d+)\s*[.\-–—]\s*(\d+)"      # 例3.1 / 例题 3-1
    r"|例\s*题?\s*(\d+)"                        # 例题3
    r"|第\s*(\d+)\s*题"                         # 第3题
    r"|(\d+)\s*[.、．]\s*(?=\S)"                # 3. / 3、 / 3．
    r")"
)

#: 解答起点在**行首**（`解 …` / `解析：…`）
SOLUTION_MARK = re.compile(r"^\s*(?:解析|解答|答案|证明|解|Answer\b|Solution\b)\s*[:：]?\s*", re.IGNORECASE)

#: 解答起点在**行中间**：`…总加速度a。解` —— 实测这就是主流形态。
#:
#: 只匹配行首会把所有解答都漏掉（第一版就是这样，5 道题的解答全部为空）。
#: 用「句末标点 + 解」作边界，避免把"求解""理解"这类词误判成标记。
SOLUTION_INLINE = re.compile(r"(?<=[。．.？！?])\s*(?:解|解析|解答|证明)\s*[:：]?\s*")

#: 这些行是**章节标题**不是题号（`一、选择题`），要排掉
SECTION_HEADER = re.compile(r"^\s*[一二三四五六七八九十]+\s*[、.．]\s*\S")

#: 小问编号 `(1)` `（1）`——不是题号
SUB_QUESTION = re.compile(r"^\s*[（(]\s*\d+\s*[)）]")

#: 题干最长多少字（超过就多半是误切，把两个题粘一起了）
MAX_STATEMENT_CHARS = 1200


@dataclass
class ProblemDraft:
    """切出来的一道题（还没落盘）。"""

    number: int
    statement: str = ""
    solution: str = ""
    pages: list[int] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    raw: str = ""

    @property
    def is_usable(self) -> bool:
        """够不够格当一道题。太短的丢弃——多半是误切或纯标题。"""
        return len(self.statement.strip()) >= 10


def _extract_number(match: re.Match) -> int | None:
    """从匹配里取出题号。`例3.1` 取 1（章.节里的节号）。"""
    groups = match.groups()
    for g in groups:
        if g:
            return int(g)
    return None


def split_pages(pages_text: list[tuple[int, str]]) -> list[ProblemDraft]:
    """把 `[(页码, 该页整文)]` 切成一道道题。

    序列约束是核心：只有编号等于"期望的下一个"（或 1）的候选起点才被接受。
    为此允许**一次跳号**——OCR 偶尔会漏掉一个题号，卡死会让后面全废。
    """
    problems: list[ProblemDraft] = []
    current: ProblemDraft | None = None
    expected = 1
    in_solution = False

    for page_no, text in pages_text:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            match = PROBLEM_START.match(stripped)
            # 章节标题与小问编号都不算题号
            if match and (SECTION_HEADER.match(stripped) or SUB_QUESTION.match(stripped)):
                match = None

            if match:
                number = _extract_number(match)
                # **序列约束**：只接受"期望的下一个"或"从 1 重新开始"，
                # 并容忍一次跳号（OCR 可能漏一个题号）。
                if number is not None and number in (expected, expected + 1, 1):
                    if current is not None:
                        problems.append(current)
                    current = ProblemDraft(number=number, pages=[page_no])
                    in_solution = False
                    expected = number + 1
                    rest = stripped[match.end():].strip()
                    if rest:
                        # **题号与解答可能在同一行**（`1. 求…总加速度a。解 …`）。
                        # 这里直接 continue 会把解答标记整条漏掉——踩过。
                        inline = SOLUTION_INLINE.search(rest)
                        if inline:
                            in_solution = True
                            current.statement = rest[:inline.start()].strip()
                            current.solution = rest[inline.end():].strip()
                        elif SOLUTION_MARK.match(rest):
                            in_solution = True
                            current.solution = SOLUTION_MARK.sub("", rest, count=1).strip()
                        else:
                            current.statement = rest
                    continue

            if current is None:
                continue
            if page_no not in current.pages:
                current.pages.append(page_no)

            if in_solution:
                current.solution = (current.solution + '\n' + stripped).strip()
                continue

            # 解答边界：先看行首，再看**行中间**（实测后者才是主流）
            # `…总加速度a。解` —— 只匹配行首会让所有解答都漏掉。
            if SOLUTION_MARK.match(stripped):
                in_solution = True
                current.solution = SOLUTION_MARK.sub("", stripped, count=1).strip()
                continue
            inline = SOLUTION_INLINE.search(current.statement + '\n' + stripped) \
                if current.statement else None
            if inline:
                joined = current.statement + '\n' + stripped
                in_solution = True
                current.statement = joined[:inline.start()].strip()
                current.solution = joined[inline.end():].strip()
                continue

            current.statement = (current.statement + '\n' + stripped).strip()

    if current is not None:
        problems.append(current)

    # 收尾：raw 留全量、丢掉太短的
    usable = []
    for p in problems:
        p.raw = f"{p.statement}\n{p.solution}".strip()
        if not p.is_usable:
            logger.info("丢弃第 %d 题：题干过短（%d 字）", p.number, len(p.statement))
            continue
        if len(p.statement) > MAX_STATEMENT_CHARS:
            logger.info("第 %d 题题干过长（%d 字），可能有误切", p.number, len(p.statement))
        usable.append(p)
    return usable


def attach_figures(problem: ProblemDraft, figures: list[dict]) -> None:
    """把插图按**所在页**归属到题。

    只按页归属，不做更细的 bbox 就近判定——实测跨页图本来就少，
    而页级归属不会把图弄丢（宁可多给，不可漏给）。
    """
    problem.figures = [f for f in figures if f.get("page") in problem.pages]


def split_pdf(pdf_path: str | Path, *, use_ocr_text: bool = True) -> list[ProblemDraft]:
    """从 PDF 切题。用文字层（扫描件则用其 OCR 层）。"""
    import pymupdf

    pages_text: list[tuple[int, str]] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i in range(doc.page_count):
            pages_text.append((i + 1, doc[i].get_text("text") or ""))
    return split_pages(pages_text)


def to_problem_payload(problem: ProblemDraft, *, problem_id: str,
                       course: str = "大学物理") -> dict:
    """转成 `data/problems/{id}/problem.json` 的结构。

    **字段与既有题库完全一致**（`content` 而不是 `statement`——
    文件格式用的是 `content`，模型字段名靠别名映射）。
    这样切出来的题能直接进 P14 建图流程，不必为"导入的题"另开一条路。
    """
    title = problem.statement.strip().splitlines()[0][:60] if problem.statement else ""
    return {
        "problem_id": problem_id,
        "title": title or f"导入题目 {problem.number}",
        "content": problem.statement.strip(),
        "knowledge_points": [],
        "difficulty": 3,
        "answer": problem.solution.strip(),
        "figures": [f.get("path", "") for f in problem.figures if f.get("path")],
        "metadata": {
            "source": "imported_exam",
            "course": course,
            "exam_number": problem.number,
            "pages": problem.pages,
        },
    }
