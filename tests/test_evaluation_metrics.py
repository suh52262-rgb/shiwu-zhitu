"""P18 指标：六个纯函数。

## 这组测试盯的两件事

1. **口径**：每个指标用哪个口径必须明确（P18 前置的兑现）
2. **"没有数据" ≠ "结果很好"**：算不出来时返回 `None`，不返回 0

第 2 条是实测踩出来的——假诊断率最初的接线把"用例没标注 `is_error`"
当成"学生是对的"，于是把一条真错误用例算成误判，报出 25% 的假诊断率。
**那个数字是假的，而且是往坏里假**——反过来（把没测的当成测得好）
就会让一份没有依据的报告看起来很漂亮。
"""
import pytest

from app.evaluation.metrics import (
    METRIC_SPECS,
    Rate,
    alternative_path_acceptance,
    diagnosis_accuracy,
    false_diagnosis_rate,
    hit_rate,
    mrr,
    node_recall,
    structure_valid_rate,
)


# ---------------------------------------------------------------- 通用语义

def test_rate_carries_denominator():
    """**分母必须一起报。** `1/1` 和 `100/100` 同为 100%，
    可信度完全不同；只报百分比是评测报告最常见的不诚实。"""
    r = Rate(1.0, 1, 1)
    assert "1/1" in r.render()
    r2 = Rate(1.0, 100, 100)
    assert "100/100" in r2.render()


def test_uncomputable_renders_as_na_not_zero():
    """**算不出来时显示 n/a，不显示 0%。**"""
    r = Rate(None, note="无负样本")
    assert not r.computable
    assert "n/a" in r.render()
    assert "0%" not in r.render()
    assert "无负样本" in r.render()


# ---------------------------------------------------------------- 节点

def test_node_recall_basic():
    r = node_recall(["S1", "S2", "S3"], {"S1", "S3"})
    assert r.value == pytest.approx(2 / 3)
    assert (r.numerator, r.denominator) == (2, 3)
    assert "S2" in r.detail[0]


def test_node_recall_without_gold_is_na():
    """用例没声明 gold 集时**算不出来**，不能报成 0% 或 100%。"""
    assert not node_recall([], {"S1"}).computable


def test_node_recall_full_and_empty():
    assert node_recall(["S1"], {"S1"}).value == 1.0
    assert node_recall(["S1"], set()).value == 0.0


# ---------------------------------------------------------------- 诊断

def test_false_diagnosis_rate_counts_only_negatives():
    """分母是**负样本**（学生其实是对的）——那才是"冤枉"的分母。"""
    labeled = [("a", True, True),     # 真错，报错 → 对
               ("b", False, False),   # 真对，没报 → 对
               ("c", True, False)]    # 真对，报了错 → **冤枉**
    r = false_diagnosis_rate(labeled)
    assert (r.numerator, r.denominator) == (1, 2)
    assert r.value == 0.5
    assert "c" in r.detail[0]


def test_false_diagnosis_rate_na_without_negatives():
    """**全是错误样本时无法计算。**

    这正是本次要补负样本的原因：没有负样本，"假诊断率"根本无从谈起，
    而报成 0% 会让人以为系统从不冤枉学生。
    """
    r = false_diagnosis_rate([("a", True, True), ("b", True, True)])
    assert not r.computable
    assert "负样本" in r.note


def test_unlabeled_cases_are_excluded_not_assumed_correct():
    """**缺标注 ≠ 学生是对的。**

    实测踩过：`exp.get("issue_type", "none")` 把"没这个键"默认成
    "没有错误"，于是真错误用例 `dg_003`（学生自述代入出错、
    只是没写 `issue_type`）被算成误判，报出 25% 假诊断率。
    """
    labeled = [("real_error", True, None),   # 未标注
               ("neg", False, False)]        # 负样本
    r = false_diagnosis_rate(labeled)
    assert (r.numerator, r.denominator) == (0, 1), "未标注的用例被算进去了"
    assert "real_error" not in str(r.detail)


def test_diagnosis_accuracy_excludes_unlabeled():
    labeled = [("a", True, True), ("b", True, None)]
    r = diagnosis_accuracy(labeled)
    assert (r.numerator, r.denominator) == (1, 1)
    assert "1 条未标注" in r.note


# ---------------------------------------------------------------- 替代路径

def test_alternative_path_acceptance():
    r = alternative_path_acceptance([("alt_001", True), ("alt_002", False)])
    assert r.value == 0.5
    assert "alt_002" in r.detail[0]


def test_alternative_path_na_without_cases():
    assert not alternative_path_acceptance([]).computable


# ---------------------------------------------------------------- 检索

def test_hit_rate():
    r = hit_rate([("q1", True), ("q2", False), ("q3", True), ("q4", True)])
    assert r.value == 0.75
    assert (r.numerator, r.denominator) == (3, 4)
    assert "q2" in r.detail[0]


def test_mrr_rewards_higher_rank():
    """排第 1 位得 1.0，第 2 位得 0.5 —— **Hit Rate 看不出这个差别**，
    而 Tutor 只把前几条喂给模型，排名直接决定学生看到什么。"""
    assert mrr([("q", 1)]).value == 1.0
    assert mrr([("q", 2)]).value == 0.5
    assert mrr([("q", 4)]).value == 0.25
    assert mrr([("q", None)]).value == 0.0


def test_mrr_mixed():
    r = mrr([("a", 1), ("b", 2), ("c", None), ("d", None)])
    assert r.value == pytest.approx((1.0 + 0.5) / 4)
    assert r.denominator == 4


def test_mrr_is_not_rendered_as_a_count_ratio():
    """**MRR 不是计数比**，报成 `46%（4/4）` 会被读成"4 条里中了 4 条"
    ——正好相反（4 条里 2 条排第 1、2 条落空才得 0.5）。"""
    r = mrr([("a", 1), ("b", None)])
    text = r.render()
    assert "50%" in text
    assert "1/2" not in text, "MRR 被渲染成了计数比"
    assert "样本" in text


# ---------------------------------------------------------------- 结构

def test_structure_valid_rate():
    r = structure_valid_rate([("g1", True, []), ("g2", False, ["no_start"])])
    assert r.value == 0.5
    assert "g2" in r.detail[0]


def test_structure_valid_rate_na_without_graphs():
    assert not structure_valid_rate([]).computable


# ---------------------------------------------------------------- 元数据

def test_every_metric_documents_its_semantics():
    """**报告里必须带口径**，否则数字无法解读（P18 前置的直接要求）。"""
    for name, (label, semantics) in METRIC_SPECS.items():
        assert label, f"{name} 缺中文名"
        assert semantics, f"{name} 缺口径说明"
