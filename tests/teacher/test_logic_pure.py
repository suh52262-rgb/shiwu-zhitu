"""纯函数测试：分类 / 聚合 / 覆盖面 / 班级 / 诊断措辞。

守的坑（审阅编号见交接包报告）：
  · #6 覆盖面用 max 少算 → 必须用精确去重值，拿不到就不说
  · #7 班级人均排名不可比 → 不得出现"相对更弱"等评价性表述
  · #5 "提示=卡住"越界 → 不得出现该类措辞
  · 班级重复成员 → 必须拒绝而不是靠字典顺序决定归属
"""
from __future__ import annotations

import json

import pytest

import teacher_logic as L


# ---------------------------------------------------------------- 分类

def test_classify_orders_specific_before_generic():
    """「电磁感应」必须归电磁学，不能被力学的单字「力」抢走。"""
    assert L.classify_discipline("电磁感应") == "电磁学"
    assert L.classify_discipline("简谐振动") == "力学"
    assert L.classify_discipline("光电效应") == "量子物理"


def test_unclassified_is_not_forced_into_a_category():
    """没命中就是「未分类」——把分类错误伪装成成功更糟。

    注意用词：随便挑中文词可能**恰好命中**关键词规则
    （例如「相干」属于光学，那么「不相干」也会命中）。
    所以这里用完全不在规则里的英文字符串。
    """
    assert L.classify_discipline("zzz_unmapped_zzz") == L.UNCLASSIFIED


def test_subfield_uses_secondary_rules_for_mechanics():
    assert L.classify_subfield("刚体定轴转动") == "刚体转动"
    assert L.classify_subfield("动量守恒") == "动量与碰撞"


# ---------------------------------------------------------------- 聚合

def test_aggregate_exposes_upper_hint_not_a_headcount():
    """聚合后的学生数只能叫 `students_upper_hint`，不能叫 `students`。

    原因：聚合行里已没有学生名单，取 max 会**少算**真实并集。
    字段名若叫 `students`，调用方几乎必然当人数用 → 就是 #6 那个 bug。
    """
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 2, "student_count": 1},
        {"knowledge_point": "冲量", "mention_count": 1, "student_count": 1},
    ])
    tot = L.discipline_totals(rows)
    assert tot, "应至少有一个学科"
    assert "students" not in tot[0], "不得再输出会被误当人数的 `students`"
    assert "students_upper_hint" in tot[0]


def test_subfield_totals_does_not_collapse_into_discipline():
    """历史上这里写错过：字典展开顺序导致领域分组塌回学科名。"""
    rows = L.normalize_rows([
        {"knowledge_point": "刚体定轴转动", "mention_count": 1, "student_count": 1},
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 1},
    ])
    names = [s["name"] for s in L.subfield_totals(rows)]
    assert "刚体转动" in names and "动量与碰撞" in names


# ---------------------------------------------------------------- #6 覆盖面

def test_coverage_uses_exact_count_not_max():
    """两个互不重叠的学生必须算 2 人（原实现用 max 算出 1 人 / 50%）。"""
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 1},
        {"knowledge_point": "电容", "mention_count": 1, "student_count": 1},
    ])
    items = L.build_diagnosis(weak_rows=rows, active_students=2,
                             covered_students=2, attempt_count=2)
    cov = [d for d in items if "覆盖面" in d["title"]]
    assert cov, "应输出覆盖面结论"
    assert "2 名（100%）" in cov[0]["text"], cov[0]["text"]


def test_coverage_omitted_when_exact_count_unavailable():
    """拿不到精确值时**宁可不说** —— 不能拿会少算的数字糊弄。"""
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 1},
        {"knowledge_point": "电容", "mention_count": 1, "student_count": 1},
    ])
    items = L.build_diagnosis(weak_rows=rows, active_students=2,
                             covered_students=None, attempt_count=2)
    assert not [d for d in items if "覆盖面" in d["title"]]


def test_coverage_inconsistency_is_reported_not_clamped():
    """分子 > 分母时必须**报矛盾**，而不是 min() 钳制掩盖。

    审阅原文：「数据矛盾报异常，**不用min钳制掩盖不同查询范围**」。
    钳制会让比率看起来正常，把"口径已经不一致"藏起来 ——
    那比显示一个荒谬的 150% 更危险（后者至少逼人去看）。
    """
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 5},
    ])
    items = L.build_diagnosis(weak_rows=rows, active_students=2,
                             covered_students=5, attempt_count=2)
    cov = [d for d in items if "覆盖面" in d["title"]]
    assert cov, "必须有一条覆盖面相关结论（矛盾报告）"
    text = cov[0]["text"]
    assert "不一致" in cov[0]["title"] or "不一致" in text, cov[0]
    #: 不得给出任何百分比（那是被钳制出来的假数字）
    assert "%" not in text, f"矛盾时不应输出百分比：{text}"
    assert "不输出覆盖率" in text or "停止输出" in text, text


def test_coverage_normal_case_still_reports_ratio():
    """正常情形仍要有覆盖率（别把矛盾处理做过头）。"""
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 2},
    ])
    items = L.build_diagnosis(weak_rows=rows, active_students=2,
                             covered_students=2, attempt_count=2)
    cov = [d for d in items if "覆盖面" in d["title"]][0]
    assert "2 名（100%）" in cov["text"]
    assert "100%" in cov["text"]


# ---------------------------------------------------------------- #5 措辞

@pytest.mark.parametrize("banned", [
    "发提示即学生卡住",
    "下发提示 = 学生卡住",
    "在某节点下发提示 = 学生在那里卡住",
    "提示整体有效",
    "提示整体偏弱",
])
def test_hint_wording_never_equates_hint_with_stuck(banned):
    """诊断文本里不得把"发提示"等同于"学生卡住"，也不得评价提示整体效果。

    主动求助、连续求助、缺推导都不能证明认知卡顿；
    `effective` 只是启发式判定，不是教学因果效果。
    """
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 5, "student_count": 2},
    ])
    stuck = [{"node": "n1", "from_hint": 4, "from_mistake": 1, "count": 5}]
    hints = {"total": 5, "effective": 4, "ineffective": 0, "undecided": 1}
    text = " ".join(d["title"] + d["text"] for d in L.build_diagnosis(
        weak_rows=rows, hint_stats=hints, stuck_nodes=stuck,
        active_students=2, covered_students=2, attempt_count=5))
    assert banned not in text


def test_diagnosis_separates_hint_and_error_nodes():
    """提示下发与错误记录必须**分开列**，口径写清楚。"""
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 3, "student_count": 2},
    ])
    stuck = [{"node": "n1", "from_hint": 3, "from_mistake": 2, "count": 5}]
    items = L.build_diagnosis(weak_rows=rows, stuck_nodes=stuck,
                             active_students=2, covered_students=2, attempt_count=5)
    node_items = [d for d in items if "节点" in d["title"]]
    assert node_items
    t = node_items[0]["text"]
    assert "提示下发较集中" in t and "错误记录较集中" in t
    assert "不表示学生在该处认知卡住" in t


# ---------------------------------------------------------------- #7 班级

def test_duplicate_members_are_rejected(tmp_path):
    """同一学生在多个班 → 拒绝，不靠字典顺序决定归属。"""
    f = tmp_path / "classes.json"
    f.write_text(json.dumps({"classes": {"A": ["s1", "s2"], "B": ["s2"]}},
                            ensure_ascii=False), encoding="utf-8")
    classes, err = L.load_classes(f)
    assert classes == {}
    assert "s2" in err


def test_broken_classes_file_does_not_raise(tmp_path):
    f = tmp_path / "classes.json"
    f.write_text("{ not json", encoding="utf-8")
    classes, err = L.load_classes(f)
    assert classes == {} and err


def test_empty_classes_returns_error(tmp_path):
    f = tmp_path / "classes.json"
    f.write_text('{"classes": {}}', encoding="utf-8")
    classes, err = L.load_classes(f)
    assert classes == {} and err


def test_class_summaries_has_no_ability_ranking():
    """班级概览只给记录分布，字段名不得暗示"人均能力"。"""
    roster = [
        {"student_id": "s1", "attempt_count": 3, "problem_attempt_count": 2,
         "weak_count": 2, "hint_count": 1},
        {"student_id": "s2", "attempt_count": 1, "problem_attempt_count": 0,
         "weak_count": 0, "hint_count": 0},
    ]
    rows = L.class_summaries(roster, {"A": ["s1"], "B": ["s2"]}, active_ids=["s1"])
    assert rows
    for r in rows:
        assert "weak_per_student" not in r, "旧的『人均薄弱点』字段不该再出现"
        assert "weak_per_student_record" in r
        assert r.get("active_count") is not None
        assert r.get("coverage") is not None


def test_compare_classes_never_says_weaker():
    """对照文案不得出现"相对更弱"这类能力评价。"""
    summaries = [
        {"name": "A", "student_count": 3, "active_count": 3, "attempt_count": 9,
         "problem_attempt_count": 6, "weak_count": 9, "hint_count": 3,
         "weak_per_student_record": 3.0, "problem_rate": 0.67, "coverage": 1.0},
        {"name": "B", "student_count": 3, "active_count": 3, "attempt_count": 3,
         "problem_attempt_count": 1, "weak_count": 2, "hint_count": 1,
         "weak_per_student_record": 0.67, "problem_rate": 0.33, "coverage": 1.0},
    ]
    text = " ".join(d["title"] + d["text"] for d in L.compare_classes(summaries))
    assert "相对更弱" not in text
    assert "更弱" not in text
    assert "不是能力" in text or "不是**能力" in text


def test_compare_classes_flags_low_coverage_as_incomparable():
    summaries = [
        {"name": "A", "student_count": 10, "active_count": 1, "attempt_count": 1,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0,
         "weak_per_student_record": 0.0, "problem_rate": 0.0, "coverage": 0.1},
        {"name": "B", "student_count": 10, "active_count": 10, "attempt_count": 20,
         "problem_attempt_count": 8, "weak_count": 12, "hint_count": 4,
         "weak_per_student_record": 1.2, "problem_rate": 0.4, "coverage": 1.0},
    ]
    text = " ".join(d["title"] + d["text"] for d in L.compare_classes(summaries))
    assert "覆盖不足" in text


def test_compare_classes_needs_two_classes():
    assert L.compare_classes([]) == []
    assert L.compare_classes([{"name": "A", "student_count": 1}]) == []


# ---------------------------------------------------------------- 范围指纹

def test_scope_fingerprint_is_deterministic():
    """指纹必须**确定性** —— 同一范围每次算出来一样。

    不能用内置 `hash()`：它受 PYTHONHASHSEED 影响，**每次进程启动都不同**，
    那样就没法用它判断"两次查询是否同一范围"。
    """
    a = L.scope_fingerprint(start="2026-10-01", end="2026-10-14",
                            student_ids=["s2", "s1"], attempt=None)
    b = L.scope_fingerprint(start="2026-10-01", end="2026-10-14",
                            student_ids=["s1", "s2"], attempt=None)
    assert a == b, "同一范围（学生集合顺序不同）指纹应相同"


def test_scope_fingerprint_changes_with_scope():
    """范围变了，指纹必须跟着变 —— 否则它证明不了任何事。"""
    base = dict(start="2026-10-01", end="2026-10-14", student_ids=["s1"])
    assert L.scope_fingerprint(**base) != L.scope_fingerprint(
        start="2026-10-01", end="2026-10-13", student_ids=["s1"])
    assert L.scope_fingerprint(**base) != L.scope_fingerprint(
        start="2026-10-01", end="2026-10-14", student_ids=["s1", "s2"])
    assert L.scope_fingerprint(**base) != L.scope_fingerprint(
        start="2026-10-01", end="2026-10-14", student_ids=["s1"], problem_id="p1")


def test_scope_fingerprint_leaks_nothing_sensitive():
    """指纹只用于标识一致性，不得含口令等敏感信息。"""
    fp = L.scope_fingerprint(start="2026-10-01", end="2026-10-14",
                             student_ids=["s1"], extra="范围A")
    assert "shiwu_demo_local" not in fp
    assert "password" not in fp.lower()


def test_diagnosis_cards_carry_scope_id():
    """每张诊断卡片都要带上同一个范围指纹（审阅要求 snapshot 标识）。"""
    rows = L.normalize_rows([
        {"knowledge_point": "动量守恒", "mention_count": 3, "student_count": 2},
    ])
    items = L.build_diagnosis(weak_rows=rows, active_students=2,
                             covered_students=2, attempt_count=5,
                             scope_label="测试范围", scope_id="RID-1")
    assert items
    assert all(x.get("scope_id") == "RID-1" for x in items), \
        [x.get("scope_id") for x in items]
    assert all(x.get("scope_label") == "测试范围" for x in items)


def test_empty_diagnosis_also_carries_scope_id():
    """**早返回分支也要盖章** —— 那些分支最容易被误读成"真的没数据"。"""
    items = L.build_diagnosis(weak_rows=[], active_students=0,
                             scope_id="RID-2", scope_label="空范围")
    assert items and all(x.get("scope_id") == "RID-2" for x in items)


def test_wrong_diagnosis_carries_scope_id():
    items = L.build_wrong_diagnosis([], active_students=0,
                                    scope_id="RID-3", scope_label="空范围")
    assert items and all(x.get("scope_id") == "RID-3" for x in items)
    rows = [{"problem_id": "p1", "student_id": "s1", "issue_types": "concept",
             "nodes": "n1", "mistake_count": 1}]
    items2 = L.build_wrong_diagnosis(rows, active_students=1,
                                     scope_id="RID-4", scope_label="有数据")
    assert items2 and all(x.get("scope_id") == "RID-4" for x in items2)


# ---------------------------------------------------------------- 时间口径不可混算

def test_lifetime_weak_scope_does_not_mix_time_frames():
    """**跨时累计的弱项不得与范围作答数拼进同一句**（审阅第二批 #6）。

    审阅原文：「个人画像是跨时累计时，**也不得与当前日期范围作答次数
    拼成同一段诊断**」。

    为什么：`weak_rows` 有两种来源 ——
      · 按范围过滤（`attempts.weak_concepts`）→ 与 attempt_count 同口径
      · 跨时累计（`student_profiles.weak_concepts`）→ **不同口径**

    后者若与"范围内作答 N 次"写在一句，读者会误解为
    "这 N 次作答产生了 M 个弱项"。**即使两个数字各自都对，这句话也是错的。**
    """
    weak = L.normalize_rows([
        {"knowledge_point": "切向加速度", "mention_count": 1, "student_count": 1},
        {"knowledge_point": "法向加速度", "mention_count": 1, "student_count": 1},
        {"knowledge_point": "动量守恒", "mention_count": 1, "student_count": 1},
    ])
    items = L.build_diagnosis(
        weak_rows=weak, active_students=1, covered_students=1,
        attempt_count=1,                      # 当前范围只 1 次作答
        scope_label="学生 s1",
        weak_scope=L.WEAK_SCOPE_LIFETIME,
    )
    #: **关键：必须拆成两条独立卡片**，而不是一句里分两句。
    #: 放在同一张卡里读者仍会把两个数字联系起来（"这 1 次作答产生了那 3 个弱项"）。
    scale_cards = [x for x in items if "规模" in x["title"] or "画像" in x["title"]]
    assert len(scale_cards) >= 2, [x["title"] for x in items[:3]]

    #: 逐卡检查：**按具体数字判断**，而不是按关键词共现。
    #: （画像卡的解释文字里会提到「这几次作答」以说明"不能这样理解"，
    #:   按关键词判会误报 —— 要判的是"数字有没有混在一起"。）
    scope_card = [c for c in scale_cards if "筛选范围" in c["title"]][0]
    img_card = [c for c in scale_cards if "历史累计画像" in c["title"]][0]

    #: 范围卡：只讲范围数字，**不得出现弱项数**
    assert "1 次作答" in scope_card["text"], scope_card["text"]
    assert "薄弱知识点" not in scope_card["text"], scope_card["text"]

    #: 画像卡：只讲累计弱项，**不得出现"范围作答数"这种陈述**
    assert "3 个薄弱知识点" in img_card["text"], img_card["text"]
    assert "1 次作答" not in img_card["text"], img_card["text"]
    assert "名学生" not in img_card["text"], img_card["text"]

    #: 画像卡必须**明确说明**它不受筛选影响、且不能与作答数相互推算
    #: （只说"口径不同"不够 —— 要给出为什么，读者才不会自行联系两个数字）
    assert "不随上方" in img_card["text"], img_card["text"]
    assert "不能相互推算" in img_card["text"], img_card["text"]
    assert "跨时间累计" in img_card["text"], img_card["text"]


def test_range_weak_scope_still_combines_in_one_sentence():
    """同口径（range）时仍可合并陈述 —— 别把上一条做过头。"""
    weak = L.normalize_rows([
        {"knowledge_point": "切向加速度", "mention_count": 2, "student_count": 2},
    ])
    items = L.build_diagnosis(
        weak_rows=weak, active_students=2, covered_students=2,
        attempt_count=5, scope_label="全部数据",
        weak_scope=L.WEAK_SCOPE_RANGE,
    )
    scale = items[0]["text"]
    assert "2 名学生、5 次作答" in scale, scale
    assert "历史画像提及 1 个知识点" in scale, scale
    assert "2 次历史提及" in scale, scale


def test_distribution_does_not_invent_shared_learning_cause():
    rows = [{'problem_id': p, 'student_id': s, 'mistake_count': 1, 'issue_types': 'calculation'}
            for p in ['p1', 'p2'] for s in ['s1', 's2']]
    text = ' '.join(item['text'] for item in L.build_wrong_diagnosis(rows))
    assert '没打牢' not in text and '不是偶发' not in text
    assert '计算步骤' in text and '不能仅凭分布' in text


def test_missing_error_records_are_not_proof_of_correctness():
    text = ' '.join(item['text'] for item in L.build_wrong_diagnosis([]))
    assert '没有产生错误' not in text
    assert '不证明学生没有做错' in text


def test_default_weak_scope_is_range():
    """默认必须是 range（保守：不声明累计时按同口径处理）。"""
    assert L.WEAK_SCOPE_RANGE == "range"
    assert L.WEAK_SCOPE_LIFETIME == "lifetime"
    weak = L.normalize_rows([
        {"knowledge_point": "x", "mention_count": 1, "student_count": 1},
    ])
    items = L.build_diagnosis(weak_rows=weak, active_students=1,
                             covered_students=1, attempt_count=1)
    assert "次作答" in items[0]["text"]


# ---------------------------------------------------------------- 条件性对照三前提

def _two_classes(rate_a=0.6, rate_b=0.2, cov=1.0):
    return [
        {"name": "A", "student_count": 3, "active_count": 3, "attempt_count": 9,
         "problem_attempt_count": int(9 * rate_a), "weak_count": 5, "hint_count": 3,
         "weak_per_student_record": 1.6, "problem_rate": rate_a, "coverage": cov},
        {"name": "B", "student_count": 3, "active_count": 3, "attempt_count": 9,
         "problem_attempt_count": int(9 * rate_b), "weak_count": 1, "hint_count": 1,
         "weak_per_student_record": 0.3, "problem_rate": rate_b, "coverage": cov},
    ]


def test_conditional_comparison_needs_all_three_preconditions():
    """**共同题目 + 共同时间 + 覆盖足够**，三者缺一不可（审阅第二批 #7）。"""
    #: 只有共同题目 → 不够
    only_p = L.compare_classes(_two_classes(), common_problems=5)
    joined = " ".join(x["title"] + x["text"] for x in only_p)
    assert "未做条件性对照" in joined, joined
    assert "共同时间窗口" in joined, joined

    #: 只有共同时间 → 不够
    only_t = L.compare_classes(_two_classes(), common_time_window="2026-10-01~10-14")
    joined_t = " ".join(x["title"] + x["text"] for x in only_t)
    assert "未做条件性对照" in joined_t, joined_t
    assert "共同题目数" in joined_t, joined_t

    #: 两者齐备 → 才给条件性对照
    full = L.compare_classes(_two_classes(), common_problems=5,
                             common_time_window="2026-10-01~10-14")
    joined_f = " ".join(x["title"] + x["text"] for x in full)
    assert "条件性对照" in joined_f, joined_f
    assert "未做条件性对照" not in joined_f, joined_f


def test_missing_precondition_is_stated_not_silently_skipped():
    """前提不足时必须**说明**，不能默默不显示。

    默默不显示会让读者以为"两班水平相当" —— 而实际是"前提不成立、不可比"。
    """
    items = L.compare_classes(_two_classes())
    joined = " ".join(x["title"] + x["text"] for x in items)
    assert "未做条件性对照" in joined, joined
    assert "不代表两个班水平相当" in joined or "尚不可比" in joined, joined


def test_low_coverage_also_counts_as_missing_precondition():
    cls = _two_classes()
    cls[0]["coverage"] = 0.1                 # A 班覆盖不足
    items = L.compare_classes(cls, common_problems=5,
                              common_time_window="2026-10-01~10-14")
    joined = " ".join(x["title"] + x["text"] for x in items)
    assert "未做条件性对照" in joined, joined
    assert "覆盖" in joined, joined


def test_compare_classes_never_says_weaker_still_holds():
    """回归：仍不得出现能力评价措辞。"""
    text = " ".join(x["title"] + x["text"]
                    for x in L.compare_classes(_two_classes(), common_problems=5,
                                               common_time_window="w"))
    assert "相对更弱" not in text and "更弱" not in text
