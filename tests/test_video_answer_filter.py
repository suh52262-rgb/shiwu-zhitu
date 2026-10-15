"""Perception descriptions must never become student answers."""
import json

from app.api.video import _snapshots_from
from scripts.drive_video_attempt import load_snapshots


def test_blank_and_failed_roi_cannot_replace_transcription(tmp_path):
    data = {"frames": [
        {"timestamp_sec": 0, "description": "blank notebook", "has_content": False},
        {"timestamp_sec": 1, "description": "hand holding a pen"},
        {"timestamp_sec": 2, "text": "invented", "has_text": False},
        {"timestamp_sec": 3, "text": "N=mg", "frame_type": "keyframe"},
        {"timestamp_sec": 3, "text": "wrong", "frame_type": "roi", "error": "timeout"},
        {"timestamp_sec": 23, "ocr_lines": ["F=ma"], "description": "formula"},
    ]}
    snapshots = _snapshots_from(data, 20)
    assert [(s["ts"], s["text"]) for s in snapshots] == [(3, "N=mg"), (23, "F=ma")]
    source = tmp_path / "results.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    assert load_snapshots(source, 20) == snapshots


def test_all_description_video_has_no_submittable_answer():
    assert _snapshots_from({"frames": [
        {"timestamp_sec": 0, "description": "blank notebook", "text": ""},
        {"timestamp_sec": 20, "description": "pen moving", "ocr_lines": []},
    ]}, 20) == []
