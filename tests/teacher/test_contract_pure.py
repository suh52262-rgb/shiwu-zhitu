"""统一只读聚合契约测试（`teacher_contract.py`）。

守的坑（审阅第三批 + 第一批 #4）：
  · 「避免把规则再复制进 SQL」→ 谓词必须有**单一来源**，不能各写一遍
  · 「分开展示**历史提及**与**模型记录的错误**」→ 两个谓词**不得互相包含**
  · 「覆盖面…独立 COUNT(DISTINCT student_id)」→ 契约里要有对应落点，
    且谓词本身必须是合法 WHERE 片段（不能含聚合函数）
"""
from __future__ import annotations

import re

import pytest

import teacher_contract as C
import teacher_logic as L


# ---------------------------------------------------------------- 结构完整性

def test_signals_have_required_fields():
    """信号条目必须写清来源、性质、是/不是 —— 否则界面无法据实说明。"""
    for key, sig in C.SIGNALS.items():
        for field in ("name", "source", "nature", "is", "is_not", "time_scope"):
            assert field in sig, f"信号 {key} 缺字段 {field}"
        assert sig["is"] and sig["is_not"], f"信号 {key} 的 is/is_not 不能为空"
        assert sig["time_scope"] in (C.SCOPE_RANGE, C.SCOPE_LIFETIME)


def test_metrics_have_required_fields():
    for key, m in C.METRICS.items():
        for field in ("name", "sql", "time_scope"):
            assert field in m, f"指标 {key} 缺字段 {field}"
        assert m["time_scope"] in (C.SCOPE_RANGE, C.SCOPE_LIFETIME), key
        assert "{where}" in m["sql"], f"指标 {key} 的 SQL 必须留 {{where}} 占位"


def test_scope_rules_cover_both_scopes():
    assert set(C.SCOPE_RULES) == {C.SCOPE_RANGE, C.SCOPE_LIFETIME}
    #: 「不得拼进同一段」这条禁止项必须在规则文本里写明
    assert "不得" in C.SCOPE_RULES[C.SCOPE_LIFETIME]


def test_scope_constants_match_logic_module():
    """契约与逻辑层的范围口径必须同义 —— 两处不一致会让界面文字与统计对不上。"""
    assert C.SCOPE_RANGE == L.WEAK_SCOPE_RANGE
    assert C.SCOPE_LIFETIME == L.WEAK_SCOPE_LIFETIME


# ---------------------------------------------------------------- 信号不得混算

def test_error_predicate_does_not_contain_weak_concepts():
    """核心回归：错误记录的判据**只看 mistakes**。

    原实现 `_has_mistakes` 把 `mistakes OR weak_concepts` 都算成"有问题"，
    导致**历史弱过的学生每道题都被算成有问题**。
    """
    p = C.sql_error_records("a")
    assert "a.mistakes" in p
    assert "weak_concepts" not in p, f"错误谓词混入了 weak_concepts：{p}"


def test_weak_predicate_does_not_contain_mistakes():
    p = C.sql_weak_mention("a")
    assert "a.weak_concepts" in p
    assert "mistakes" not in p, f"历史提及谓词混入了 mistakes：{p}"


def test_predicates_respect_alias():
    for fn in (C.sql_error_records, C.sql_weak_mention, C.sql_covered_students):
        p = fn("x")
        assert "a." not in p, f"{fn.__name__} 未替换别名：{p}"
        assert "x." in p


def test_predicates_are_valid_where_fragments():
    """谓词片段不得含聚合函数 —— 否则拼进 WHERE 就是非法 SQL。"""
    for fn in (C.sql_error_records, C.sql_weak_mention, C.sql_covered_students):
        p = fn("a")
        assert not re.search(r"\b(count|sum|avg|min|max)\s*\(", p, re.I), \
            f"{fn.__name__} 返回了聚合表达式，不能作 WHERE 片段：{p}"


def test_covered_students_predicate_is_pure_filter():
    """覆盖面谓词同 weak 过滤条件；聚合方式由调用方负责。"""
    assert C.sql_covered_students("a") == C.sql_weak_mention("a")


# ---------------------------------------------------------------- 指标口径

def test_covered_students_metric_uses_count_distinct():
    """审阅：「独立 COUNT(DISTINCT student_id)」——
    不能拿"各知识点学生数的最大值"代替（实测会少算）。"""
    sql = C.METRICS["covered_students"]["sql"]
    assert "count(DISTINCT" in sql.replace("COUNT", "count")
    assert "max(" not in sql.lower(), "不得用 max 代替并集计数"


def test_problem_attempts_and_weak_flagged_are_separate_metrics():
    """两项必须分开存在，且不得相加 —— 口径不同。"""
    pa = C.METRICS["problem_attempts"]["sql"]
    wf = C.METRICS["weak_flagged"]["sql"]
    assert "mistakes" in pa and "weak_concepts" not in pa
    assert "weak_concepts" in wf and "mistakes" not in wf
    assert "必须分开" in C.METRICS["weak_flagged"]["note"] or \
           "分开" in C.METRICS["weak_flagged"]["note"]


def test_hint_undecided_excluded_from_denominator():
    """未判定必须单列、不计入分母 —— 当成"无效"会系统性低估。"""
    note = C.METRICS["hint_undecided"]["note"]
    assert "不计入分母" in note


def test_hint_effectiveness_is_heuristic_not_causal():
    """`effective` 的说明必须点明是启发式判定，不是教学因果效果。"""
    note = C.METRICS["hint_effective"]["note"]
    assert "启发式" in note and "不是教学因果效果" in note


def test_signals_declare_time_scope_correctly():
    """历史提及是跨时累计；错误记录随范围 —— 与界面两卡分离一致。"""
    assert C.SIGNALS[C.SIGNAL_HISTORICAL]["time_scope"] == C.SCOPE_LIFETIME
    assert C.SIGNALS[C.SIGNAL_ERROR_RECORD]["time_scope"] == C.SCOPE_RANGE


# ---------------------------------------------------------------- 指纹

def test_fingerprint_is_stable_and_hex():
    fp = C.contract_fingerprint()
    assert fp == C.contract_fingerprint(), "同一进程内必须稳定"
    assert re.fullmatch(r"[0-9a-f]{12}", fp), fp


def test_fingerprint_changes_when_metric_sql_changes(monkeypatch):
    """口径一变指纹必变 —— 否则"按哪版口径算的"就答不出来。"""
    before = C.contract_fingerprint()
    patched = dict(C.METRICS)
    patched["attempts"] = {**C.METRICS["attempts"], "sql": "count(*) /* changed */"}
    monkeypatch.setattr(C, "METRICS", patched)
    assert C.contract_fingerprint() != before


def test_fingerprint_changes_when_signal_added(monkeypatch):
    before = C.contract_fingerprint()
    patched = dict(C.SIGNALS)
    patched["brand_new_signal"] = {"time_scope": C.SCOPE_RANGE}
    monkeypatch.setattr(C, "SIGNALS", patched)
    assert C.contract_fingerprint() != before


def test_contract_version_present():
    assert isinstance(C.CONTRACT_VERSION, str) and C.CONTRACT_VERSION


# ---------------------------------------------------------------- 单一来源

def test_teacher_data_uses_contract_predicates():
    """数据层必须**引用**契约谓词，而不是各写一遍。

    （审阅：「避免把规则再复制进 SQL」。这条测试防的是"以后又有人就地写一份"。）
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2]
           / "frontend" / "teacher_data.py").read_text(encoding="utf-8")
    assert "from teacher_contract import" in src, "数据层未引用契约模块"
    assert "sql_error_records" in src and "sql_weak_mention" in src

    #: 就地硬写 `json_array_*(COALESCE(a.mistakes|weak_concepts ...))` 的判据
    #: 不应再出现 —— 谓词/取值只能由 `teacher_contract.py` 定义一次。
    #: （⚠️ 正则必须覆盖**所有**三种写法，否则只挡住 `json_array_length` 这一种，
    #:   而 `json_array_elements` / `..._text` 会从缝里漏过去。）
    inlined = re.findall(
        r"json_array_(?:length|elements(?:_text)?)\s*\(\s*COALESCE\s*\(\s*\w+\.(mistakes|weak_concepts)",
        src)
    assert not inlined, f"数据层出现就地硬写的判据（应改用契约 helper）：{inlined}"


@pytest.mark.parametrize("name", ["sql_error_records", "sql_weak_mention", "sql_covered_students"])
def test_predicate_functions_exported(name):
    assert callable(getattr(C, name))
