"""P24 教师端 + P25 回放。

## P24 测什么
教师端要回答的是**班级层面**的问题（"这届学生在哪卡住"），
所以全是聚合。这里盯两件事：
  · 提示效果的**三态分开数**——把"未判定"混进分母，会因"没测到"而显得效果差
  · 键名如实（`issue_type` 就叫 `issue_type`，别叫 `knowledge_point`）

## P25 测什么
回放是**给现场用的**，所以状态机的每条边都必须确定：
pause 中单步不能自动续播、reset 要停住、非法倍速要拒绝。
这些"小行为"在演示现场出问题最难救。
"""
import inspect

import pytest


# ================================================================ P24

def test_teacher_routes_registered():
    """4 条路由都要在。"""
    from app.main import app

    paths = {getattr(r, "path", "") for r in app.routes}
    # 路由可能挂在子 router 上，用 openapi 更可靠
    schema = app.openapi()
    got = set(schema.get("paths", {}))
    for want in ("/teacher/students/{student_id}",
                 "/teacher/attempts/{attempt_id}",
                 "/teacher/problems/{problem_id}/stats",
                 "/teacher/knowledge-points/stats"):
        assert want in got, f"缺路由 {want}（现有：{sorted(got)}）"


def test_teacher_uses_group_by():
    """**P24 的实质是聚合**——全项目此前没有一条 GROUP BY。"""
    from app.api import teacher

    src = inspect.getsource(teacher)
    assert "group_by" in src.lower(), "教师端没有一条聚合"
    assert "func.count()" in src


def test_hint_effect_three_states_are_separate():
    """**未判定单列，不混进分母。**

    教师端只报"有效率"的话，`None`（未判定）会被算进分母，
    于是**因为没测到而显得提示效果差**——这是个会误导教学决策的数字。
    """
    from app.api import teacher

    src = inspect.getsource(teacher)
    assert "undecided" in src, "没有单列未判定"
    assert "effective" in src and "ineffective" in src
    # 分母必须是可判定条数
    assert "/ decided" in src, "分母不是可判定条数"


def test_issue_type_is_named_honestly():
    """**键名如实**：`frequent_issues` 里装的是错误类型，不是知识点。

    任务书示例写的是 `knowledge_point`，但那是示例——叫错名字会让 C 端
    按知识点去解读错误类型。知识点维度另有 `/teacher/knowledge-points/stats`。
    """
    from app.api import teacher

    src = inspect.getsource(teacher.teacher_problem_stats)
    assert '"issue_type"' in src, "错误类型被错标成 knowledge_point"


def test_teacher_degrades_without_db():
    """数据库不可用时教师端返回空结果，**不抛异常给前端**。"""
    from app.api import teacher

    assert teacher._rows is not None
    assert teacher._rows_raw is not None
    # 查询函数必须自带兜底
    assert "except Exception" in inspect.getsource(teacher._rows)


# ================================================================ P25

@pytest.fixture()
def rec(tmp_path, monkeypatch):
    """一段三帧的假录制。"""
    import json

    from app.api import demo

    (tmp_path / "case_x.json").write_text(json.dumps({
        "case_id": "case_x", "title": "测试案例", "problem_id": "phy_x",
        "recorded_at": "2026-01-01T00:00:00Z", "source": "test",
        "steps": [
            {"index": 1, "student_input": "a", "visited_nodes": ["S1"],
             "timeline": [{"n": 1}]},
            {"index": 2, "student_input": "b", "visited_nodes": ["S1", "S2"],
             "timeline": [{"n": 2}]},
            {"index": 3, "student_input": "c", "visited_nodes": ["S3"],
             "timeline": [{"n": 3}]},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(demo, "RECORDINGS_DIR", tmp_path)
    return demo


def test_playback_starts_playing(rec):
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    assert s.state == rec.STATE_IDLE
    s.start()
    assert s.state == rec.STATE_PLAYING
    assert s.cursor == 0, "start 不该自动播放第一帧"


def test_next_while_paused_stays_paused(rec):
    """**现场演示常是"暂停着一步步讲"。**

    单步之后自动续播会让讲的人失去节奏——所以 `next` 不改变 playing/paused。
    """
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    s.start()
    s.pause()
    s.next()
    assert s.cursor == 1
    assert s.state == rec.STATE_PAUSED, "暂停中单步后自动续播了"


def test_reset_stops_at_beginning(rec):
    """`reset` 回到开头**并停住**——不是立刻又播起来。"""
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    s.start()
    s.next()
    s.next()
    s.reset()
    assert s.cursor == 0
    assert s.state == rec.STATE_PAUSED


def test_finishes_at_the_end(rec):
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    s.start()
    for _ in range(5):
        s.next()
    assert s.cursor == 3, "游标越过了末尾"
    assert s.state == rec.STATE_FINISHED


def test_speed_only_allows_three_steps(rec):
    """**只开放 x1/x2/x4**——任意倍速会让"这一帧该停多久"难以预期。"""
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    for ok in rec.ALLOWED_SPEEDS:
        s.set_speed(ok)
    for bad in (0.5, 3.0, 10.0):
        with pytest.raises(ValueError):
            s.set_speed(bad)


def test_view_grows_cumulatively(rec):
    """**回放的意义**：Timeline 与节点路径随播放逐渐长出来。

    一上来就把整段结果摊开，那就不是"回放"而是"展示结果"了。
    """
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    s.start()
    assert s.view()["timeline"] == []
    s.next()
    assert len(s.view()["timeline"]) == 1
    s.next()
    v = s.view()
    assert len(v["timeline"]) == 2
    assert v["visited_nodes"] == ["S1", "S2"], "节点路径没有累积"


def test_view_records_provenance(rec):
    """**现场要能回答"这是真跑出来的吗"。**

    回放的是录制数据而不是实时结果，所以来源必须如实标明——
    `recorded_at` + `source`。这不是 metadata，是诚实性要求。
    """
    s = rec.PlaybackSession(rec.load_recording("case_x"))
    v = s.view()
    assert v["source"] == "test"
    assert v["recorded_at"]


def test_missing_recording_lists_available(rec):
    """找不到录制时给出的错误要**可操作**（列出有哪些）。"""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        rec.load_recording("nope")
    assert ei.value.status_code == 404
    assert "case_x" in ei.value.detail, "错误信息没有列出可用录制"


def test_playback_does_not_call_any_model():
    """**回放一次模型都不调**——这是它能在断网现场跑的前提。

    源码里不该出现模型/检索调用。
    """
    from app.api import demo

    src = inspect.getsource(demo)
    for forbidden in ("get_llm(", "call_vision", "retrieve_knowledge("):
        assert forbidden not in src, f"回放模块里出现了 {forbidden}，现场会依赖外部服务"
