"""试卷切题（`app/rag/exam_split.py`）。

## 这里断言的核心是「不会误切」

题号识别最容易出的错不是漏切，而是**把解答内部的编号当成新题号**——
一旦误切，题库里就多出一道半截题，事后清理比先测出来麻烦得多。
"""
import pytest

from app.rag.exam_split import (
    ProblemDraft,
    split_pages,
    to_problem_payload,
)


def _page(text: str, page: int = 1):
    return [(page, text)]


# ---------------------------------------------------------------- 题号识别

def test_splits_numbered_problems():
    problems = split_pages(_page(
        "1. 第一题的题干内容足够长。\n"
        "2. 第二题的题干内容足够长。\n"
        "3. 第三题的题干内容足够长。"
    ))
    assert [p.number for p in problems] == [1, 2, 3]


def test_does_not_split_on_sub_questions():
    """`(1)` `（2）` 是小问不是题号——按它们切会把一道题碎成好几道。"""
    problems = split_pages(_page(
        "1. 求下列各量。\n"
        "(1) 第一小问的内容。\n"
        "（2）第二小问的内容。\n"
        "2. 第二道题的题干内容足够长。"
    ))
    assert len(problems) == 2, [p.number for p in problems]
    assert "第一小问" in problems[0].statement
    assert "第二小问" in problems[0].statement


def test_does_not_split_on_section_headers():
    """`一、选择题` 是章节标题不是题号。"""
    problems = split_pages(_page(
        "一、选择题\n"
        "1. 第一题的内容足够长。\n"
        "二、填空题\n"
        "2. 第二题的内容足够长。"
    ))
    assert [p.number for p in problems] == [1, 2]


def test_sequence_constraint_rejects_out_of_sequence_numbers():
    """**核心防线**：解答里出现的、不在期望序列上的编号不该被当成新题号。

    期望 2 的时候冒出个 `7.`（解答里的公式编号、行号之类），应当被挡住。

    ⚠️ **这条防线有明确的边界**：它挡得住"跳到序列外"的编号，
    **挡不住恰好接上序列的连续编号**——若解答里正好出现 `3.` `4.`
    而下一道真题是 `5.`，任何基于编号的算法都无法区分（编号本身没有区别）。
    真正的兜底是 `--dry-run`：导入前先看一眼切得对不对。
    """
    problems = split_pages(_page(
        "1. 第一题的题干内容足够长，长到不会被当成误切丢掉。\n"
        "解 计算过程如下。\n"
        "7. 这里的 7. 不在期望序列上，不是新题。\n"
        "9. 这个也不是。\n"
        "2. 第二题的题干内容也足够长，长到能留下来。"
    ))
    assert [p.number for p in problems] == [1, 2], [p.number for p in problems]


def test_tolerates_one_skipped_number():
    """OCR 偶尔漏一个题号——卡死会让后面全废，所以容忍一次跳号。"""
    problems = split_pages(_page(
        "1. 第一题的内容足够长。\n"
        "3. 第三题的内容足够长（第二题被 OCR 漏掉了）。"
    ))
    assert [p.number for p in problems] == [1, 3]


def test_recognizes_example_style_headers():
    problems = split_pages(_page(
        "例1.6 吊扇转动的题干内容足够长。\n"
        "例2.2 滑轮连接体的题干内容足够长。"
    ))
    assert len(problems) == 2, [p.number for p in problems]


# ---------------------------------------------------------------- 题干/解答边界

def test_solution_marker_at_line_start():
    problems = split_pages(_page(
        "1. 题干内容足够长足够长。\n"
        "解 这是解答的开头。"
    ))
    assert "解答的开头" in problems[0].solution
    assert "解答的开头" not in problems[0].statement


def test_solution_marker_mid_line():
    """**实测这才是主流形态**：`…总加速度a。解` 紧跟在题干末尾。

    第一版只匹配行首，结果 5 道题的解答全部为空。
    """
    problems = split_pages(_page(
        "1. 求关闭电源后的角加速度、切向加速度和总加速度a。解 吊扇翼尖P原来的角速度"
    ))
    p = problems[0]
    assert p.statement.endswith("总加速度a。"), p.statement
    assert "吊扇翼尖P原来的角速度" in p.solution
    assert "吊扇翼尖P原来的角速度" not in p.statement


def test_no_solution_leaves_empty_string():
    problems = split_pages(_page("1. 这道题只有题干没有解答，内容足够长。"))
    assert problems[0].solution == ""
    assert problems[0].statement


def test_problem_without_solution_is_still_usable():
    """没解答的题也要能用——P14 的提示词本身就支持"未提供参考答案"。"""
    problems = split_pages(_page("1. 只有题干的题目，内容长度足够当一道题。"))
    assert len(problems) == 1
    assert problems[0].is_usable


# ---------------------------------------------------------------- 多页与丢弃

def test_tracks_pages_across_pages():
    problems = split_pages([(1, "1. 第一题的内容足够长。"), (2, "第二题续。"),
                            (3, "2. 第二题的内容足够长。")])
    assert problems[0].pages == [1, 2]
    assert problems[1].pages == [3]


def test_short_fragments_are_dropped():
    """太短的丢弃——多半是误切或纯标题。"""
    problems = split_pages(_page(
        "1. 短。\n"
        "2. 第二题的内容长度足够当一道题。"
    ))
    assert [p.number for p in problems] == [2], [p.number for p in problems]


# ---------------------------------------------------------------- 落盘结构

def test_payload_matches_existing_problem_format():
    """字段必须与既有题库一致，否则导入的题进不了 P14 流程。"""
    p = ProblemDraft(number=3, statement="题干内容足够长足够长。",
                     solution="解答内容。", pages=[1, 2])
    payload = to_problem_payload(p, problem_id="exam_003")
    # 既有 problem.json 用的是 content 而不是 statement
    assert "content" in payload and "statement" not in payload
    for key in ("problem_id", "title", "content", "answer", "figures",
                "metadata", "knowledge_points", "difficulty"):
        assert key in payload, f"缺少字段 {key}"
    assert payload["metadata"]["pages"] == [1, 2]
