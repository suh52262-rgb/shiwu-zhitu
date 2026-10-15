"""受控词表的测试。

## 它要解决的问题
推荐的判据是「画像知识点 ∩ 题库知识点」。而实测：
· 现有 8 道题 72 条知识点里 **6 组是同义分叉**
· `phy_003` 的自动图还把**"做题技巧"标成了知识点**（`代数化简`、`递推关系`）

**那些分叉不是意外，是"每题各自写"这个做法的必然结果。**
不清根的话，**加新题（如教材例题 466 道）会批量重演**。

## 设计取舍
· 词表载入失败**不抛异常**（返回空表）—— 它是治理工具，不是运行前提
· 新概念**可以加**，但必须有意为之（`allow_new_concepts`）
· **"明确不承认"的标签任何时候都挡住** —— 那是已经做过的判断，不该悄悄放回来
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(str(Path(__file__).resolve().parents[1] / 'backend'))))

from app.services.knowledge_vocabulary import (  # noqa: E402
    VOCAB_PATH,
    OutOfVocabulary,
    aliases,
    assert_in_vocabulary,
    canonical,
    dropped,
    load_vocabulary,
    validate,
    vocabulary,
)


# ================================================================ 词表本身

def test_vocabulary_file_exists_and_is_not_empty():
    assert VOCAB_PATH.is_file(), "受控词表文件不存在"
    assert len(vocabulary()) >= 40, "词表太少了，像是没生成"


def test_vocabulary_is_sorted_and_unique():
    v = vocabulary()
    assert v == sorted(v), "词表应当有序（便于人看）"
    assert len(v) == len(set(v))


def test_aliases_point_to_real_concepts():
    """**别名必须指向表里真实存在的规范名** —— 否则归位会归到一个不存在的概念。"""
    v = set(vocabulary())
    for a, c in aliases().items():
        assert c in v, f"别名 {a!r} 指向的 {c!r} 不在词表里"


def test_dropped_are_not_also_concepts():
    """**"不承认"的东西不该同时在词表里** —— 那自相矛盾。"""
    assert not (set(dropped()) & set(vocabulary()))


# ================================================================ 归位

def test_canonical_resolves_aliases():
    assert canonical("机械能守恒") == "机械能守恒条件"
    assert canonical("简谐运动振幅") == "简谐运动"


def test_canonical_passes_through_unknown():
    """表外的**逐字返回** —— 不硬造一个"看起来像"的。"""
    assert canonical("量子纠缠") == "量子纠缠"
    assert canonical("") == ""


# ================================================================ 校验

def test_validate_accepts_vocabulary_items():
    v = vocabulary()[:3]
    r = validate(v)
    assert r["ok"] is True and r["unknown"] == []


def test_validate_flags_dropped():
    """**"明确不承认"的标签要被单独标出**（不是混在 unknown 里）。"""
    d = dropped()
    assert d, '源码中的受控词表必须包含用于回归的排除项'
    r = validate([d[0]])
    assert r["ok"] is False
    assert r["dropped"] == [d[0]]
    assert r["unknown"] == []


def test_validate_flags_unknown():
    r = validate(["一个全新概念"])
    assert r["ok"] is False
    assert r["unknown"] == ["一个全新概念"]


def test_validate_accepts_aliases_as_known():
    """别名算"已知" —— 归位后就是规范名。"""
    r = validate(["机械能守恒"])
    assert r["ok"] is True
    assert "机械能守恒条件" in r["known"]


# ================================================================ 断言

def test_assert_raises_on_unknown():
    with pytest.raises(OutOfVocabulary) as e:
        assert_in_vocabulary(["一个全新概念"])
    # 报错信息要说清**怎么办**
    msg = str(e.value)
    assert "aliases" in msg and "dropped" in msg, "没说清三条处置路径"


def test_assert_allows_new_when_explicit():
    """**新概念可以加，但要显式放行** —— 不该成为常态，但不该堵死。"""
    assert_in_vocabulary(["一个全新概念"], allow_new=True)


def test_assert_still_blocks_dropped_even_when_allow_new():
    """⚠️ **`allow_new` 也不能把"明确不承认"的放回来。**

    那是一个已经做过的人工判断（"代数化简不是知识点"），
    放行新概念的意思是"可能有新概念"，不是"推翻既有判断"。
    """
    d = dropped()
    assert d, '源码中的受控词表必须包含用于回归的排除项'
    with pytest.raises(OutOfVocabulary):
        assert_in_vocabulary([d[0]], allow_new=True)


def test_broken_vocabulary_does_not_crash():
    """**词表坏了不该让题库打不开** —— 它是治理工具，不是运行前提。"""
    r = validate(["随便什么"], path="D:/不存在的路径.json")
    assert r["ok"] is True, "词表不可用时应放行，而不是拦住"
    assert load_vocabulary("D:/不存在的路径.json") == {}


# ================================================================ 接线

def test_write_problem_validates_knowledge_points():
    """**防"写了没接"** —— 校验必须在写入口。"""
    import inspect

    from app.services import problem_writer as pw

    src = inspect.getsource(pw.write_problem)
    assert "assert_in_vocabulary" in src, "write_problem 没做词表校验"
    # 节点级也要过 —— 推荐用的是题目级 + 节点级的并集
    assert "knowledge_points" in src


def test_write_problem_still_has_the_overwrite_guard():
    """**别把已有的护栏挤掉了。**"""
    import inspect

    from app.services import problem_writer as pw

    src = inspect.getsource(pw.write_problem)
    assert "force" in src and "已存在" in src


def test_existing_problems_all_pass():
    """**对着真实数据自检**：现有 8 道题的知识点必须全在词表里。

    这条是防"词表与题库脱节" —— 一旦脱节，加新题时的报警就成了噪声。
    """
    from app.services.problem_repository import get_problem_repository

    repo = get_problem_repository()
    ids = repo.list_problem_ids()
    assert ids, '独立测试题库必须存在，不得跳过词表接线检查'
    for pid in ids:
        p = repo.get_problem(pid)
        kps = list(p.knowledge_points or [])
        g = repo.get_graph(pid)
        for n in getattr(g, "nodes", None) or []:
            kps += list(n.knowledge_points or [])
        r = validate(kps)
        assert r["ok"], f"{pid} 有表外知识点：unknown={r['unknown']} dropped={r['dropped']}"
