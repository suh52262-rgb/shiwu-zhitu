"""导出的产物必须报**实际用的模型**。

## 为什么单独测
`run_pipeline` 的返回值我修过了，但 **`results.json` 和 `report.md` 才是
事后被翻出来看的证据**——它们一直写着提示值。

**一份说错话的证据比没有证据更糟**：它会让人以为某段是用
`qwen3-vl-plus` 跑的，而实际是 `deepseek-flash`。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.vision import service as sv  # noqa: E402


def _fake_records():
    return [
        {"frame": "a.jpg", "timestamp_sec": 0.0, "type": "keyframe",
         "model_used": "deepseek-flash", "text": "a"},
        {"frame": "b.jpg", "timestamp_sec": 2.0, "type": "keyframe",
         "model_used": "deepseek-flash", "text": "a b"},
        {"frame": "c.jpg", "timestamp_sec": 4.0, "type": "roi",
         "model_used": "glm-4.6v", "text": "b"},
    ]


def test_results_json_reports_actual_models(tmp_path, monkeypatch):
    """`results.json` 要报**实际用到的模型集合**，不是提示值。"""
    monkeypatch.setattr(sv, "_write_change_scores", lambda *a, **k: None)
    sv.export_results(Path("v.mp4"), _fake_records(), [],
                      tmp_path, stats={"probes": 1}, model="qwen3-vl-plus")
    d = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert d["models_used"] == ["deepseek-flash", "glm-4.6v"],\
        "没有如实汇总实际模型"
    assert d["model_hint"] == "qwen3-vl-plus", "提示值应另存"
    assert "model" not in d, "旧的『模型』字段还在——它会说谎"


def test_results_json_records_mixed_models(tmp_path, monkeypatch):
    """**同一轮里换过模型也要看得出来**（候选链可能中途切换）。"""
    monkeypatch.setattr(sv, "_write_change_scores", lambda *a, **k: None)
    sv.export_results(Path("v.mp4"), _fake_records(), [], tmp_path)
    d = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(d["models_used"]) == 2, "混合模型被压成一个了"


def test_report_md_reports_actual_model(tmp_path, monkeypatch):
    """`report.md` 是**给人读的**，更不能用提示值。"""
    monkeypatch.setattr(sv, "_write_change_scores", lambda *a, **k: None)
    sv.export_results(Path("v.mp4"), _fake_records(), [],
                      tmp_path, model="qwen3-vl-plus")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "deepseek-flash" in md, "报告里没有实际模型"
    assert "实际使用的模型" in md


def test_report_notes_the_hint_when_it_differs(tmp_path, monkeypatch):
    """提示值与实际不同时要**写出来**——那正是误导最容易发生的地方。"""
    monkeypatch.setattr(sv, "_write_change_scores", lambda *a, **k: None)
    sv.export_results(Path("v.mp4"), _fake_records(), [],
                      tmp_path, model="qwen3-vl-plus")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "qwen3-vl-plus" in md, "没写出提示值，无法看出两者不同"


def test_records_without_model_used_do_not_crash(tmp_path, monkeypatch):
    """老记录没有 `model_used` 字段时要能跑（向后兼容）。"""
    monkeypatch.setattr(sv, "_write_change_scores", lambda *a, **k: None)
    old = [{"frame": "a.jpg", "timestamp_sec": 0.0, "type": "keyframe", "text": "x"}]
    sv.export_results(Path("v.mp4"), old, [], tmp_path)
    d = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert d["models_used"] == []
