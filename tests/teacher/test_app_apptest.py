"""AppTest 离线故障探针：降级页、空班级、界面文案。

守的坑（审阅「第一批」）：
  · #1 只导入 MODE_DB 却在故障分支引用 MODE_API → 数据库不可用时
       `NameError: name 'MODE_API' is not defined`，而 README 承诺的正是
       "故障时能启动降级"。所以**故障路径必须被测试**。
  #2 班级配置缺失/损坏时选项只有「未分班」，默认值却是「全部班级」
       → `StreamlitDefaultNotInOptionsError`，整页打不开。

这些用例**不连真实数据库**：用不可达地址触发降级路径，
用临时损坏的班级文件触发空状态路径。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str((Path(__file__).resolve().parents[2] / "frontend" / "teacher_app.py"))

#: 不可达地址：保证走降级分支，不受本机是否有数据库影响。
UNREACHABLE_DB = "postgresql://u:p@127.0.0.1:59999/nodb"
UNREACHABLE_API = "http://127.0.0.1:59998"


def _texts(at: AppTest) -> list[str]:
    """抓出界面上所有可见文本（用于断言文案）。"""
    out: list[str] = []
    for kind in ("caption", "markdown", "warning", "error", "info",
                 "success", "title", "subheader", "metric", "code"):
        for el in getattr(at, kind, []) or []:
            v = (getattr(el, "value", None) or getattr(el, "label", None)
                 or getattr(el, "body", None))
            if v:
                out.append(str(v))
    return out


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TEACHER_DATABASE_URL", "TEACHER_BACKEND_URL",
                 "TEACHER_CLASSES_FILE", "TEACHER_CONFIG_FILE"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- #1 降级路径

def test_page_renders_when_db_and_api_unreachable(monkeypatch):
    """**核心回归**：数据库与后端都不可用时，页面必须能开（降级），不崩。

    原实现在这里抛 NameError（MODE_API 未导入），
    于是"故障时能启动降级"的承诺不成立。
    """
    monkeypatch.setenv("TEACHER_DATABASE_URL", UNREACHABLE_DB)
    monkeypatch.setenv("TEACHER_BACKEND_URL", UNREACHABLE_API)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    joined = " ".join(_texts(at))
    assert "不可用" in joined, "应出现明确的降级提示"


def test_page_says_filters_unavailable_in_degraded_mode(monkeypatch):
    monkeypatch.setenv("TEACHER_DATABASE_URL", UNREACHABLE_DB)
    monkeypatch.setenv("TEACHER_BACKEND_URL", UNREACHABLE_API)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    joined = " ".join(_texts(at))
    #: 不能假装筛选用得了 —— 否则教师误以为看到的是"最近 14 天"，实际是全量。
    assert "筛选不可用" in joined or "仅支持" in joined


def test_degraded_page_shows_config_source(monkeypatch):
    """降级时要能看出"它到底连到哪了"，便于排障。"""
    monkeypatch.setenv("TEACHER_DATABASE_URL", UNREACHABLE_DB)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    joined = " ".join(_texts(at))
    assert "59999" in joined, "连接配置里应回显实际连接的地址"
    assert "***" in joined, "口令必须打码"


# ---------------------------------------------------------------- #2 班级配置

def test_broken_classes_file_does_not_crash(monkeypatch, tmp_path):
    """坏 JSON 的班级文件不得让整页崩（原为默认值不在选项里）。"""
    f = tmp_path / "broken.json"
    f.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    joined = " ".join(_texts(at))
    assert "班级配置" in joined, "应明确提示班级配置有问题"


def test_empty_classes_does_not_crash(monkeypatch, tmp_path):
    f = tmp_path / "empty.json"
    f.write_text('{"classes": {}}', encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_missing_classes_file_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(tmp_path / "nope.json"))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_duplicate_member_classes_does_not_crash(monkeypatch, tmp_path):
    """重复成员会被 load_classes 拒绝 → 页面仍需能开且给出原因。"""
    f = tmp_path / "dup.json"
    f.write_text(json.dumps({"classes": {"A": ["s1", "s2"], "B": ["s2"]}},
                            ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    joined = " ".join(_texts(at))
    assert "多个班级" in joined


# ---------------------------------------------------------------- #3 空班级

def test_empty_class_yields_zero_students_and_no_data_leak(monkeypatch, tmp_path, teacher_db_fake):
    """**核心回归 #3**：选一个没有成员的班级 → 0 名学生，且不得退化成全库。

    原实现把空列表当"不筛选"，页面写"0 名学生"却显示所有班的数据。
    """
    fake, calls = teacher_db_fake
    fake.default_roster = [
        {'student_id': 's1', 'display_name': 's1', 'attempt_count': 2,
         'problem_attempt_count': 0, 'weak_count': 0, 'hint_count': 0},
        {'student_id': 's2', 'display_name': 's2', 'attempt_count': 3,
         'problem_attempt_count': 0, 'weak_count': 0, 'hint_count': 0},
    ]
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"classes": {"空班": ["根本不存在的人"]}},
                            ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("TEACHER_CLASSES_FILE", str(f))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception

    ms = [b for b in at.multiselect if b.key == "class_filter"]
    assert ms, 'DB替身必须渲染班级选择器，不能跳过空班级隔离检查'
    ms[0].set_value(["空班"]).run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("覆盖学生") == "0", metrics
    assert metrics.get("作答次数") == "0", metrics
    joined = " ".join(_texts(at))
    assert "没有学生" in joined, "应显式说明空集合导致无数据"
    assert [kw for method, kw in calls if method == 'counts'][-1]['student_ids'] == []


# ---------------------------------------------------------------- 文案口径

@pytest.mark.parametrize("banned", [
    "发提示即学生卡住", "提示整体有效", "提示整体偏弱",
    "学生在那里卡住", "相对更弱", "人均薄弱点", "有问题的作答",
])
def test_ui_never_shows_overreaching_wording(monkeypatch, banned):
    """界面不得出现已判定越界/不可比的表述。"""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    joined = " ".join(_texts(at))
    assert banned not in joined, f"界面仍出现：{banned}"


ALL_VIEWS = ["① 知识点分布", "② 平行检索", "③ 错题分析", "④ 班级与个体"]


@pytest.mark.parametrize("view", ALL_VIEWS)
def test_every_view_avoids_ability_ranking_wording(monkeypatch, view):
    """**逐视图**检查：不得出现班级能力评价类措辞。

    为什么逐视图而不是只看默认页：默认只渲染视图 ①，
    视图 ④ 的文案（"哪个班更弱"）就曾经这样漏过去 ——
    实测发现后才补上这个用例。
    """
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    radios = [r for r in at.radio if r.key == "teacher_view"]
    assert radios, '视图切换器必须渲染，不能跳过逐视图文案检查'
    radios[0].set_value(view).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    joined = " ".join(_texts(at))
    for banned in ("更弱", "哪个班更差", "问题最多", "最需要关注"):
        assert banned not in joined, f"{view} 仍出现：{banned}"


def test_ui_states_signal_provenance(monkeypatch):
    """界面必须说明「历史提及」的性质，避免被读成"当前确认的错误"。"""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    joined = " ".join(_texts(at))
    assert "口径" in joined or "历史提及" in joined
