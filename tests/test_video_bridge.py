"""服务层 records → LearningEvent 的**映射**。

⚠️ 本文件**显式关闭稳定期**（`settle_window=0`）：这里的用例只关心
"快照怎么变事件"，而稳定期是另一个关注点（见 `test_video_settle.py`）。
用例只有 2~3 条记录，凑不满 3 帧窗口，开着就什么都不产出——
那不是映射错了，是**两个关注点混在一个测试里**。

原始说明：

任务书 4.3：**"主 Agent 不直接处理原始视频，而只消费 LearningEvent。"**

测试重点：
1. **同刻的整帧 + ROI 必须合并**（否则每写一笔都变成两条事件）
2. **ROI 文字优先**（它是针对变化区域的定向读数）
3. **有 error 的记录不伪造内容**，但要留下缺口痕迹
"""
import pytest

from app.models.event import EVENT_TYPES
from app.vision import bridge as br

FULL = {"frame_type": "keyframe", "timestamp_sec": 0.0, "text": "N=mg",
        "description": "纸上写着一个公式"}
ROI = {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 0.0,
       "text": "N=mg"}


def _full(ts, text, kind="keyframe"):
    return {"frame_type": kind, "timestamp_sec": ts, "text": text}


# ---------------------------------------------------------------- 合并

def test_full_and_roi_at_same_ts_merge_into_one():
    """**核心**：整帧 + 它的 ROI 是同一时刻的两种读数，不是两个时刻。

    不合并的话，时间线上每写一笔都会变成两条。
    """
    merged = br.merge_by_timestamp([FULL, ROI])
    assert len(merged) == 1, f"同刻记录没合并：{merged}"
    assert merged[0]["n"] == 2
    assert merged[0]["ts"] == 0.0


def test_roi_text_wins_over_full_frame():
    """ROI 是针对变化区域的定向读数，比整帧更准。"""
    full = {"frame_type": "keyframe", "timestamp_sec": 0.0,
            "text": "整帧粗略读数"}
    roi = {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 0.0,
           "text": "N=mg cosθ"}
    merged = br.merge_by_timestamp([full, roi])
    assert merged[0]["text"] == "N=mg cosθ"


def test_roi_after_full_still_wins():
    """顺序无关——ROI 在后也要赢。"""
    roi = {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 1.0,
           "text": "精细"}
    full = {"frame_type": "keyframe", "timestamp_sec": 1.0, "text": "粗略"}
    assert br.merge_by_timestamp([roi, full])[0]["text"] == "精细"


def test_frame_type_from_full_not_roi():
    """整帧记录提供时刻性质；ROI 记录的 type 是 'roi'，不代表时刻性质。"""
    merged = br.merge_by_timestamp([
        {"frame_type": "static_rep", "timestamp_sec": 5.0, "text": "x"},
        {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 5.0, "text": "x"},
    ])
    assert merged[0]["type"] == "static_rep"


def test_output_sorted_by_ts():
    recs = [_full(10.0, "b"), _full(0.0, "a"), _full(5.0, "c")]
    assert [m["ts"] for m in br.merge_by_timestamp(recs)] == [0.0, 5.0, 10.0]


# ---------------------------------------------------------------- 取文字

def test_text_sources_in_priority_order():
    assert br._text_of({"text": "T"}) == "T"
    assert br._text_of({"description": "D"}) == "D"
    assert br._text_of({"ocr_lines": ["a", "b"]}) == "a\nb"
    assert br._text_of({}) == ""
    # text 优于 description
    assert br._text_of({"text": "T", "description": "D"}) == "T"


# ---------------------------------------------------------------- 端到端

def test_records_become_events():
    recs = [
        _full(0.0, "N=mg"),
        _full(10.0, "N=mg"),
        _full(20.0, "N=mg cosθ"),
    ]
    evs = br.records_to_learning_events(recs, attempt_id="a1",
                                        student_id="s1", problem_id="phy_2", settle_window=0)
    assert [e.event_type for e in evs] == ["start", "modify"]
    assert evs[1].content == "将 N=mg 改为 N=mg cosθ"
    assert evs[1].metadata["student_id"] == "s1"
    assert evs[1].metadata["problem_id"] == "phy_2"


def test_full_plus_roi_does_not_double_events():
    """**整帧 + ROI 不该让事件翻倍**——这是合并的实际收益。"""
    recs = [
        {"frame_type": "keyframe", "timestamp_sec": 0.0, "text": "N=mg"},
        {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 0.0, "text": "N=mg"},
        {"frame_type": "keyframe", "timestamp_sec": 9.0, "text": "N=mg cosθ"},
        {"frame_type": "roi", "roi_tag": "auto", "timestamp_sec": 9.0,
         "text": "N=mg cosθ"},
    ]
    evs = br.records_to_learning_events(recs, settle_window=0)
    assert [e.event_type for e in evs] == ["start", "modify"], \
        f"事件翻倍了：{[(e.event_type, e.content) for e in evs]}"


def test_error_records_do_not_fabricate_content():
    """**识别失败的帧不编内容。**

    编一个出来比缺失更糟——它会被下游当成真的。
    """
    recs = [
        _full(0.0, "a"),
        {"frame_type": "keyframe", "timestamp_sec": 5.0, "text": "",
         "error": "APIConnectionError: 连接失败"},
        _full(10.0, "a\nb"),
    ]
    evs = br.records_to_learning_events(recs, settle_window=0)
    assert all("连接失败" not in e.content for e in evs)
    # 缺口要留痕
    assert "gaps" in evs[-1].metadata
    assert any("第 2 帧" in g for g in evs[-1].metadata["gaps"])


def test_all_event_types_legal():
    recs = [_full(0.0, "a"), _full(1.0, "a\nb"), _full(2.0, "c"),
            {"frame_type": "static_rep", "timestamp_sec": 60.0, "text": "c"}]
    for e in br.records_to_learning_events(recs, settle_window=0):
        assert e.event_type in EVENT_TYPES


def test_empty_and_garbage_input():
    assert br.records_to_learning_events([], settle_window=0) == []
    assert br.records_to_learning_events([None, "x", 42], settle_window=0) == []


def test_missing_timestamp_defaults_to_zero():
    evs = br.records_to_learning_events([{"frame_type": "keyframe", "text": "a"}], settle_window=0)
    assert evs and evs[0].timestamp == pytest.approx(0.0)
