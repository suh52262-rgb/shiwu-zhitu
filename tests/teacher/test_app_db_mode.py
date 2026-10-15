"""AppTest **数据库模式替身**测试（审阅 T7）。

审阅指出：`test_empty_class_yields_zero_students_and_no_data_leak` 因依赖
真实数据库而被 `skip`，于是"空班级端到端隔离"从未被验收。要补的是
**DB 模式替身交互**——用一个可控的数据替身驱动 AppTest 走数据库分支，
端到端验证：

  · 空班级      → 0 名学生、**不泄漏全库**（数据层收到的必须是空列表而非 None）
  · 重复成员    → 明确错误提示（与数据库在线状态无关）
  · 未分班学生  → 正确归入「未分班」
  · 范围过滤    → 起止日期正确传给数据层

不放宽断言、不增加 skip。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import teacher_data as TD

APP = str((Path(__file__).resolve().parents[2] / "frontend" / "teacher_app.py"))

from teacher_test_fakes import FakeTeacherData, RECORD


@pytest.fixture(autouse=True)
def _env(teacher_db_fake):
    yield


def _make_app():
    return AppTest.from_file(APP, default_timeout=300)


def _texts(at: AppTest) -> list[str]:
    out: list[str] = []
    for kind in ("caption", "markdown", "warning", "error", "info",
                 "success", "title", "subheader", "metric", "code"):
        for el in getattr(at, kind, []) or []:
            v = (getattr(el, "value", None) or getattr(el, "label", None)
                 or getattr(el, "body", None))
            if v:
                out.append(str(v))
    return out


# ---------------------------------------------------------------- 空班级

def test_empty_class_yields_zero_students_and_no_data_leak_db_mode(monkeypatch, tmp_path):
    """**审阅 T7 核心**：选一个没有成员的班级 → 0 名学生，且**不泄漏全库**。

    替身数据库里有 s1、s2 两名真实学生；班级「空班」指向一个不存在的人。
    选「空班」后，数据层收到的 `student_ids` 必须是**空列表**（触发 FALSE、
    查不到任何数据），而**不是** None（None = 全体 = 泄漏）。
    """
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"classes": {"空班": ["根本不存在的人"]}},
                            ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    FakeTeacherData.default_roster = [
        {"student_id": "s1", "display_name": "s1", "attempt_count": 2,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
        {"student_id": "s2", "display_name": "s2", "attempt_count": 3,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
    ]

    at = _make_app()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    ms = [b for b in at.multiselect if b.key == "class_filter"]
    assert ms, "DB 模式下班级选择器必须渲染（不再 skip）"
    ms[0].set_value(["空班"]).run()
    assert not at.exception, [str(e.value) for e in at.exception]

    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("覆盖学生") == "0", metrics
    assert metrics.get("作答次数") == "0", metrics

    joined = " ".join(_texts(at))
    assert "没有学生" in joined, "应显式说明空集合导致无数据"

    #: 关键：选空班后，数据层收到的 student_ids 必须是 []（空），不是 None。
    empty_calls = [kw for m, kw in RECORD if m in ("weak_ranking", "counts")
                   and kw.get("student_ids") == []]
    assert empty_calls, "空班级必须把 student_ids=[] 传给数据层（而非 None=全体）"
    #: 选空班后**最后一次** counts 调用（反映最终状态）必须是空列表，
    #: 不得退化成 None（全体）。
    last_counts = [kw for m, kw in RECORD if m == "counts"][-1]
    assert last_counts.get("student_ids") == [], "选空班后数据层不得收到 None（全体）"


# ---------------------------------------------------------------- 重复成员

def test_duplicate_member_classes_shows_error_db_mode(monkeypatch, tmp_path):
    """重复成员在 DB 模式下也要给出「多个班级」错误（不因数据库在线而隐藏）。"""
    f = tmp_path / "dup.json"
    f.write_text(json.dumps({"classes": {"A": ["s1", "s2"], "B": ["s2"]}},
                            ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    FakeTeacherData.default_roster = [
        {"student_id": "s1", "display_name": "s1", "attempt_count": 1,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
        {"student_id": "s2", "display_name": "s2", "attempt_count": 1,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
    ]
    at = _make_app()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    joined = " ".join(_texts(at))
    assert "多个班级" in joined


# ---------------------------------------------------------------- 未分班

def test_unclassified_students_db_mode(monkeypatch, tmp_path):
    """不在任何班级里的学生，应可被归入「未分班」查看（不会凭空消失）。"""
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"classes": {"甲班": ["s1"]}}, ensure_ascii=False),
                 encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    FakeTeacherData.default_roster = [
        {"student_id": "s1", "display_name": "s1", "attempt_count": 1,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
        {"student_id": "s2", "display_name": "s2", "attempt_count": 2,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
    ]
    at = _make_app()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    #: 班级选择器里应有「未分班」选项（s2 属于它）
    ms = [b for b in at.multiselect if b.key == "class_filter"]
    assert ms
    opts = list(ms[0].options)
    assert "未分班" in opts, f"未分班选项缺失：{opts}"


# ---------------------------------------------------------------- 范围过滤

def test_range_filter_passes_dates_to_data_layer(monkeypatch, tmp_path):
    """选「最近 14 天」时，起止日期必须传给数据层（不能退化成全量）。"""
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"classes": {"甲班": ["s1"]}}, ensure_ascii=False),
                 encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    FakeTeacherData.default_roster = [
        {"student_id": "s1", "display_name": "s1", "attempt_count": 1,
         "problem_attempt_count": 0, "weak_count": 0, "hint_count": 0},
    ]
    at = _make_app()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    radios = [r for r in at.radio if r.key == "scope_mode"]
    assert radios, "DB 模式下范围选择器必须渲染"
    radios[0].set_value("最近 14 天").run()
    assert not at.exception, [str(e.value) for e in at.exception]

    #: 切到"最近 14 天"后，weak_ranking 应收到非空的 start。
    ranged = [kw for m, kw in RECORD if m == "weak_ranking" and kw.get("start")]
    assert ranged, "选最近 14 天时数据层必须收到 start（而非全量）"
