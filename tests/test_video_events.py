"""逐帧描述 → LearningEvent。

评审 P-3 指出的缺口：
> 逐帧描述是"**快照**"，而 LearningEvent 是"**变化事件**"

本模块做那个转换。测试重点盯三件事：
1. **快照→变化** 的映射对不对（write/modify/erase/start/idle）
2. **不伪造**（取不到文字时不编内容，只留缺口痕迹）
3. **复用既有 Schema**（不新造一套；student_id/problem_id 进 metadata）
"""
import inspect

import pytest

from app.models.event import EVENT_TYPES, LearningEvent
from app.vision import events as ev


def _rec(ts, text, kind="keyframe"):
    return {"ts": ts, "text": text, "type": kind}


# ---------------------------------------------------------------- 基本映射

def test_snapshots_become_change_events():
    """**核心语义**：连续快照里"没变"的不该产出事件。

    快照可以重复，变化不可重复——这是本模块存在的理由。
    """
    recs = [
        _rec(0.0, "N=mg"),
        _rec(5.0, "N=mg"),                      # 没变 → 无事件
        _rec(10.0, "N=mg cosθ"),                # 改了 → modify
    ]
    got = ev.frames_to_learning_events(recs, attempt_id="a1")
    assert [e.event_type for e in got] == ["start", "modify"]
    assert got[1].content == "将 N=mg 改为 N=mg cosθ"


def test_first_frame_is_start():
    got = ev.frames_to_learning_events([_rec(0.0, "mgh=1/2mv2")])
    assert len(got) == 1
    assert got[0].event_type == "start"
    assert "mgh=1/2mv2" in got[0].content


def test_added_line_is_write():
    got = ev.frames_to_learning_events([_rec(0.0, "第一行"), _rec(3.0, "第一行\n第二行")])
    assert got[1].event_type == "write"
    assert "第二行" in got[1].content


def test_removed_line_is_erase():
    got = ev.frames_to_learning_events([_rec(0.0, "第一行\n第二行"), _rec(3.0, "第一行")])
    assert got[1].event_type == "erase"
    assert "第二行" in got[1].content


def test_static_frame_without_change_yields_idle():
    """静止补帧 → `idle`。这是"学生停下来了"的信号，不是噪声。"""
    got = ev.frames_to_learning_events(
        [_rec(0.0, "x=1"), _rec(60.0, "x=1", kind="static_rep")])
    assert [e.event_type for e in got] == ["start", "idle"]


def test_unchanged_non_static_frame_yields_nothing():
    got = ev.frames_to_learning_events(
        [_rec(0.0, "x=1"), _rec(2.0, "x=1", kind="keyframe")])
    assert len(got) == 1


def test_all_event_types_are_legal():
    """产出的事件类型必须在 Schema 允许的集合里。"""
    recs = [_rec(0.0, "a"), _rec(1.0, "a\nb"), _rec(2.0, "c"),
            _rec(3.0, ""), _rec(60.0, "c", kind="static_rep")]
    for e in ev.frames_to_learning_events(recs):
        assert e.event_type in EVENT_TYPES, f"非法事件类型 {e.event_type}"


# ---------------------------------------------------------------- 诚实降级

def test_missing_text_does_not_fabricate():
    """**取不到文字时不编内容。**

    宁可少一条事件 + 留缺口痕迹，也不写一个看起来完整的 write——
    伪造出来的作答记录比缺失更糟：**它会被下游当成真的**。
    """
    recs = [_rec(0.0, "a"), _rec(5.0, ""), _rec(10.0, "a\nb")]
    got = ev.frames_to_learning_events(recs)
    assert len(got) == 2, "缺文字的那帧不该产出事件"
    assert all("第 2 帧" not in e.content for e in got)
    # **缺口要有痕迹**，否则失败就不可见（评审 P-2 批评过这点）
    assert "gaps" in got[-1].metadata
    assert "第 2 帧" in got[-1].metadata["gaps"][0]


def test_all_frames_missing_text_yields_empty():
    got = ev.frames_to_learning_events([_rec(0.0, ""), _rec(5.0, "")])
    assert got == []


def test_empty_input():
    assert ev.frames_to_learning_events([]) == []


def test_out_of_order_input_is_sorted():
    """乱序输入不该产出时间颠倒的事件顺序。"""
    got = ev.frames_to_learning_events(
        [_rec(10.0, "a\nb"), _rec(0.0, "a")])
    assert [e.timestamp for e in got] == sorted(e.timestamp for e in got)
    assert got[0].event_type == "start"


def test_fallback_text_fields():
    """VL 输出里文字可能落在 `text` / `ocr` / `description` 任一处。"""
    assert ev.frames_to_learning_events([{"ts": 0.0, "ocr": "OCR结果"}])[0]
    assert ev.frames_to_learning_events([{"ts": 0.0, "description": "描述"}])[0]


# ---------------------------------------------------------------- 会话绑定

def test_session_binding_goes_into_metadata():
    """**复用既有 Schema**：`LearningEvent` 没有 `student_id`/`problem_id`，
    评审要求补上——放 `metadata`（Schema 注释写明它是预留扩展字段），
    而不是改 Schema 加字段。"""
    got = ev.frames_to_learning_events([_rec(0.0, "a")], attempt_id="att-1",
                                       student_id="s-9", problem_id="p-3")
    assert got[0].attempt_id == "att-1"
    assert got[0].metadata["student_id"] == "s-9"
    assert got[0].metadata["problem_id"] == "p-3"
    assert got[0].metadata["source"] == "video"


def test_timestamp_is_relative_to_t0():
    """Schema 注释：时间戳是"相对 attempt 开始的秒数"。"""
    got = ev.frames_to_learning_events([_rec(100.0, "a"), _rec(130.0, "a\nb")],
                                       t0=100.0)
    assert got[0].timestamp == pytest.approx(0.0)
    assert got[1].timestamp == pytest.approx(30.0)


def test_negative_timestamp_clamped():
    """t0 晚于帧时间时不能产出负时间戳（Schema 语义是"开始之后"）。"""
    got = ev.frames_to_learning_events([_rec(5.0, "a")], t0=100.0)
    assert got[0].timestamp >= 0.0


# ---------------------------------------------------------------- 噪声过滤

def test_clock_and_separator_filtered_out():
    """屏幕录制里时钟会持续变化，**不滤掉时间线会被灌满假 write**。"""
    assert ev.normalize_lines("12:34") == []
    assert ev.normalize_lines("2026/10/04") == []
    assert ev.normalize_lines("=====") == []
    assert ev.normalize_lines("N=mg\n12:35\nx=1") == ["N=mg", "x=1"]


def test_clock_change_alone_is_not_an_event():
    """只有时钟在走 → 没有学生动作 → 不该产出事件。"""
    recs = [_rec(0.0, "N=mg\n12:34"), _rec(60.0, "N=mg\n12:35")]
    got = ev.frames_to_learning_events(recs)
    assert len(got) == 1, f"时钟变化被当成了学生动作：{[e.content for e in got]}"


# ---------------------------------------------------------------- 产出形态

def test_returns_learning_event_instances():
    got = ev.frames_to_learning_events([_rec(0.0, "a")])
    assert all(isinstance(e, LearningEvent) for e in got)


def test_timeline_view():
    got = ev.frames_to_learning_events([_rec(0.0, "a"), _rec(5.0, "a\nb")])
    tl = ev.to_timeline(got)
    assert tl[0]["type"] == "start" and tl[1]["type"] == "write"
    assert all("t" in x and "content" in x for x in tl)


def test_reuses_existing_schema_not_a_new_one():
    """**不许新造一套事件模型**——否则主 Agent 要处理两种协议。"""
    src = inspect.getsource(ev)
    assert "from app.models.event import LearningEvent" in src
    assert "class LearningEvent" not in src, "新造了 LearningEvent"
    # 产出必须能被既有 Schema 直接消费
    got = ev.frames_to_learning_events([_rec(0.0, "a")])
    LearningEvent.model_validate(got[0].model_dump())
