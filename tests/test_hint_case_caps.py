"""Hint 用例的 `max_len` 必须是**健全性上界**，不能落进实测分布里。

## 为什么单独一条

`ht_003` 曾经间歇性失败：`max_len: 800`，而同一 L5 提示跑 6 次得
411/478/553/729/751/768 字（极差 **87%**），加上更早观测到的 634/813/824，
全距 **411~824**。**800 正好落在分布中间。**

那是 **flaky 测试**，不是提示质量问题——各次内容都是完整推理链。
所以这条测试盯的是：**别再有人把上限调回分布里去。**
"""
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = EVALUATION / "datasets"

#: 各级提示的**实测**长度分布上界（来自 ht_003 的测量，见 datasets 里的
#: `max_len_basis`）。硬上限必须高于它，否则就会间歇性失败。
OBSERVED_MAX = {"ht_003": 824}


def _cases():
    p = DATASETS / "hint_cases.json"
    if not p.is_file():
        p = EVALUATION / "hint_cases.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_dataset_is_reachable():
    d = _cases()
    assert d.get("cases"), "找不到 hint 用例"


@pytest.mark.parametrize("case_id,observed", list(OBSERVED_MAX.items()))
def test_max_len_sits_above_observed_distribution(case_id, observed):
    """**硬上限必须高于实测分布。**

    否则模型偶尔啰嗦就会失败——而那与提示质量无关，
    只会让人误以为系统退化。
    """
    case = next((c for c in _cases()["cases"] if c["id"] == case_id), None)
    assert case is not None, f"用例 {case_id} 不见了"
    cap = case["expected"].get("max_len")
    assert cap is not None, f"{case_id} 没有 max_len"
    assert cap > observed, (
        f"{case_id} 的 max_len={cap} 落在实测分布（上界 {observed}）之内，"
        "会间歇性失败。要么调高，要么去改提示词真正约束长度。"
    )


def test_measured_basis_is_recorded():
    """**设定依据要写在数据里。**

    一个没有依据的数字，下一个人只会看到"能过"或"不能过"，
    不知道该往哪调、也不知道原来的数是怎么来的。
    """
    case = next(c for c in _cases()["cases"] if c["id"] == "ht_003")
    basis = case["expected"].get("max_len_basis", "")
    assert basis, "ht_003 的 max_len 没有记录依据"
    assert "实测" in basis and "87%" in basis, "依据里没有实测数据"


def test_description_explains_sanity_bound_semantics():
    """数据集说明里要写清 `max_len` 是**健全性上界**而非产品契约。

    这两者混为一谈会导致两种错误：把上界当契约（于是不敢调），
    或者当契约去改产品行为（于是为测试而改系统）。
    """
    desc = _cases().get("description", "")
    assert "健全性上界" in desc, "没有说明 max_len 的语义"
