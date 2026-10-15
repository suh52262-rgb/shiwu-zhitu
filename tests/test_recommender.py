"""题目推荐的纯逻辑测试。

## 为什么单独测
推荐是**给人看的判断**，写错了不会报错，只会**推一道无关的题** ——
学生会以为系统在乱推。所以每条排序规则都要钉住。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.recommender import (  # noqa: E402
    REASON_EASIER,
    REASON_NEW,
    REASON_WEAK,
    explain,
    recommend_problems,
    score_problem,
)


def _p(pid, kps, diff=1, title=""):
    return {"problem_id": pid, "title": title or pid,
            "knowledge_points": kps, "difficulty": diff}


PROBLEMS = [
    _p("p_mom", ["动量定理", "冲量"], 3),
    _p("p_spring", ["机械能守恒", "弹簧弹性势能"], 4),
    _p("p_rot", ["转动定律", "转动惯量"], 5),
    _p("p_easy", ["动量定理"], 1),
]


# ---------------------------------------------------------------- 打分

def test_score_counts_overlap():
    n, hit = score_problem(["动量定理", "冲量", "牛顿第二定律"], ["动量定理"])
    assert n == 1 and hit == ["动量定理"]


def test_score_normalizes_fullwidth_parens_and_spaces():
    """**全半角括号、空格不该让同一个知识点对不上。**

    题库里两种写法都有（`动量定理（冲量—动量定理）`），
    不归一化的话会**静默漏掉命中** —— 不报错，只是推荐变差。
    """
    n, _ = score_problem(["动量定理（冲量—动量定理）"], ["动量定理(冲量—动量定理)"])
    assert n == 1
    n2, _ = score_problem(["转动定律 "], ["转动定律"])
    assert n2 == 1


def test_score_ignores_blanks():
    n, _ = score_problem(["", "  ", "冲量"], ["", "冲量"])
    assert n == 1


# ---------------------------------------------------------------- 排序

def test_more_overlap_ranks_first():
    got = recommend_problems(PROBLEMS, ["动量定理", "冲量"], limit=4)
    assert got[0]["problem_id"] in ("p_mom", "p_easy"), got
    assert got[0]["score"] == 2          # p_mom 命中 2 条


def test_excludes_the_problem_just_done():
    """**刚做完的那道不能再推。**"""
    got = recommend_problems(PROBLEMS, ["动量定理"], exclude_problem_id="p_mom")
    assert "p_mom" not in [r["problem_id"] for r in got]


def test_done_problems_are_not_recommended():
    got = recommend_problems(PROBLEMS, ["动量定理"],
                             done_problem_ids={"p_mom", "p_easy"})
    assert not ({"p_mom", "p_easy"} & {r["problem_id"] for r in got})


def test_easier_first_on_tie():
    """**同分时先易后难。**

    刚被卡住的学生，再推一道更难的只会更挫败。
    """
    got = recommend_problems(PROBLEMS, ["动量定理"], limit=4)
    same = [r for r in got if r["score"] == 1]
    assert [r["difficulty"] for r in same] == sorted(r["difficulty"] for r in same)


def test_stable_order():
    """同样的输入必须给同样的顺序（不能依赖字典顺序）。"""
    a = recommend_problems(PROBLEMS, ["动量定理"], limit=4)
    b = recommend_problems(PROBLEMS, ["动量定理"], limit=4)
    assert [r["problem_id"] for r in a] == [r["problem_id"] for r in b]


def test_limit_is_respected():
    assert len(recommend_problems(PROBLEMS, ["动量定理"], limit=2)) == 2
    assert recommend_problems(PROBLEMS, ["动量定理"], limit=0) == []


# ---------------------------------------------------------------- 冷启动

def test_cold_start_returns_something():
    """**学生还没有薄弱点时不能返回空。** 一片空白还不如推几道没做过的。"""
    got = recommend_problems(PROBLEMS, [])
    assert got, "冷启动返回空了"
    assert all(r["reason"] == REASON_NEW for r in got)


def test_cold_start_prefers_easiest():
    got = recommend_problems(PROBLEMS, [])
    assert got[0]["difficulty"] == 1, "冷启动没先推最简单的"


def test_cold_start_does_not_claim_weak_hit():
    """**没命中就不能说"命中薄弱知识点"** —— 说了就是骗人。"""
    got = recommend_problems(PROBLEMS, [])
    assert all(r["reason"] != REASON_WEAK for r in got)
    assert all("命中" not in explain(r) for r in got)


def test_all_done_returns_empty_not_error():
    got = recommend_problems(PROBLEMS, ["动量定理"],
                             done_problem_ids={p["problem_id"] for p in PROBLEMS})
    assert got == []


# ---------------------------------------------------------------- 健壮性

def test_missing_difficulty_defaults_to_one():
    """**题库是数据，缺一条字段不该让整个推荐挂掉。**"""
    got = recommend_problems([{"problem_id": "x", "knowledge_points": ["动量定理"]}],
                             ["动量定理"])
    assert got and got[0]["difficulty"] == 1


def test_garbage_input_does_not_crash():
    assert recommend_problems([], ["x"]) == []
    assert recommend_problems(None, None) == []
    got = recommend_problems([{"problem_id": "a", "knowledge_points": None,
                               "difficulty": "不是数字"}], ["动量定理"])
    assert got[0]["difficulty"] == 1


def test_explain_is_specific_when_matched():
    """理由要**具体**（列出命中的知识点），不能是套话。"""
    got = recommend_problems(PROBLEMS, ["动量定理", "冲量"])
    top = got[0]
    assert top["reason"] == REASON_WEAK
    t = explain(top)
    assert "动量定理" in t or "冲量" in t


def test_reason_easier_when_weak_known_but_no_overlap():
    got = recommend_problems(PROBLEMS, ["孤立知识点"])
    assert all(r["reason"] == REASON_EASIER for r in got)
    assert "同章节" in explain(got[0])
