"""媒体端点的离线行为回归；上传文件及导出物均隔离到 tmp_path。"""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.api import attempts, video
from app.services import problem_repository, review_builder
from app.vision import service


@pytest.fixture
def media(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "runs" / "uploads")
    monkeypatch.setattr(problem_repository, "get_problem_repository",
                        lambda: SimpleNamespace(get_problem=lambda _: object()))
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_: None)
    monkeypatch.setattr(video, "_snapshots_from", lambda *_: [
        {"ts": 0.0, "text": "step 1"}, {"ts": 20.0, "text": "step 2"},
        {"ts": 40.0, "text": "step 3"}])
    coverage = {"truncated": True, "covered_until_sec": 40.0,
                "video_total_sec": 100.0, "failed_frame_count": 1,
                "failed_frames": [{"timestamp_sec": 30.0, "error": "timeout"}]}

    def pipeline(_path, out, *_args, **_kwargs):
        out.mkdir(parents=True)
        (out / "results.json").write_text(json.dumps({"frames": []}))
        return {"coverage": coverage}

    monkeypatch.setattr(service, "run_pipeline", pipeline)
    monkeypatch.setattr(video, "analyze_attempt", lambda _: SimpleNamespace(
        model_dump=lambda: {"attempt_id": "new", "finished": False}))
    return coverage


def call_video(raw=b"video", **kwargs):
    return video.analyze_video(UploadFile(filename="student.mp4", file=io.BytesIO(raw)),
                               problem_id="p", student_id="s", every_sec=20,
                               attempt_id="", request_hint=False, **kwargs)


@pytest.mark.parametrize("failure,status", [
    ("empty", 400), ("oversize", 413), ("write", 400), ("pipeline", 502),
    ("missing", 502), ("json", 502), ("snapshots", 422), ("review", None)])
def test_upload_removed_on_all_exits(media, monkeypatch, failure, status):
    if failure == "oversize":
        monkeypatch.setattr(video, "MAX_UPLOAD_BYTES", 2)
    elif failure == "write":
        original = Path.open
        def broken(path, *args, **kwargs):
            if path.parent == video.UPLOAD_DIR:
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", broken)
    elif failure in {"pipeline", "missing", "json"}:
        def pipeline(_path, out, *_args, **_kwargs):
            if failure == "pipeline":
                raise RuntimeError("vision failed")
            if failure == "json":
                out.mkdir(parents=True)
                (out / "results.json").write_text("invalid")
            return {}
        monkeypatch.setattr(service, "run_pipeline", pipeline)
    elif failure == "snapshots":
        monkeypatch.setattr(video, "_snapshots_from", lambda *_: [])
    elif failure == "review":
        def fail(*_):
            raise RuntimeError("review failed")
        monkeypatch.setattr(review_builder, "review_trajectory", fail)
    if status:
        with pytest.raises(HTTPException) as error:
            call_video(b"" if failure == "empty" else b"video")
        assert error.value.status_code == status
    else:
        assert call_video(include_review=True)["review"] is None
    assert not list(video.UPLOAD_DIR.glob("*"))


@pytest.mark.parametrize("fail", [False, True])
def test_delete_failure_is_truthful_and_preserves_error(media, monkeypatch, caplog, fail):
    original = Path.unlink
    def unlink(path, *args, **kwargs):
        if path.parent == video.UPLOAD_DIR:
            raise PermissionError("file busy")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    if fail:
        def pipeline(*_args, **_kwargs):
            raise ValueError("original failure")
        monkeypatch.setattr(service, "run_pipeline", pipeline)
        with pytest.raises(HTTPException, match="original failure"):
            call_video()
    else:
        assert call_video()["upload_removed"] is False
    assert "file busy" in caplog.text
    assert len(list(video.UPLOAD_DIR.glob("*"))) == 1


def test_completed_stops_and_passes_coverage(media, monkeypatch):
    calls = []
    def analyze(req):
        calls.append(req)
        return SimpleNamespace(model_dump=lambda: {"attempt_id": "new", "finished": True})
    monkeypatch.setattr(video, "analyze_attempt", analyze)
    response = call_video()
    assert len(calls) == 1
    assert response["teaching_stop_reason"] == "completed"
    assert response["submitted"] == 1
    assert response["remaining_snapshots"] == 2
    assert response["last_submitted_ts"] == 0.0
    assert response["coverage"] == media
    assert response["upload_removed"] is True
    assert (response["media_type"], response["student_id"], response["problem_id"]) == ("video", "s", "p")


def test_partial_failure_counts_only_success(media, monkeypatch):
    calls = []
    def analyze(req):
        calls.append(req)
        if len(calls) == 2:
            raise HTTPException(409, "finished concurrently")
        return SimpleNamespace(model_dump=lambda: {"attempt_id": "new", "finished": False})
    monkeypatch.setattr(video, "analyze_attempt", analyze)
    response = call_video()
    assert response["teaching_stop_reason"] == "analysis_error"
    assert response["submitted"] == 1
    assert response["remaining_snapshots"] == 2
    assert response["last_submitted_ts"] == 0.0
    assert "error" in response["rounds"][1]
    assert response["upload_removed"] is True


def test_first_round_conflict_preserves_status_and_removes_upload(media, monkeypatch):
    def rejected(_):
        raise HTTPException(409, "finished concurrently")
    monkeypatch.setattr(video, "analyze_attempt", rejected)
    with pytest.raises(HTTPException) as error:
        call_video()
    assert error.value.status_code == 409
    assert not list(video.UPLOAD_DIR.glob("*"))


def test_failed_ocr_records_never_become_teaching_snapshots():
    assert video._snapshots_from({"frames": [
        {"timestamp_sec": 0, "text": "unreliable", "error": "timeout"}
    ]}, 20) == []


def test_failed_roi_does_not_override_successful_full_frame():
    snapshots = video._snapshots_from({"frames": [
        {"timestamp_sec": 0, "text": "valid full frame", "frame_type": "keyframe"},
        {"timestamp_sec": 0, "text": "unreliable ROI", "roi_tag": "writing",
         "error": "timeout"}
    ]}, 20)
    assert len(snapshots) == 1
    assert snapshots[0]["text"] == "valid full frame"
    assert snapshots[0]["error"] is None


@pytest.mark.parametrize("kind", ["image", "video"])
def test_invalid_target_rejected_before_vision(media, monkeypatch, kind):
    monkeypatch.setattr(problem_repository, "get_problem_repository",
                        lambda: SimpleNamespace(get_problem=lambda _: None))
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid request reached vision")
    monkeypatch.setattr(service, "call_vision", forbidden)
    monkeypatch.setattr(service, "run_pipeline", forbidden)
    with pytest.raises(HTTPException) as error:
        if kind == "video":
            call_video()
        else:
            video.analyze_image(UploadFile(file=io.BytesIO(b"image")), "p", "s", "", False)
    assert error.value.status_code == 404
    assert not video.UPLOAD_DIR.exists()


def test_image_identity_and_http_error(media, monkeypatch):
    monkeypatch.setattr(service, "call_vision", lambda *_args, **_kwargs: ('{"text":"step"}', {}))
    monkeypatch.setattr(attempts, "analyze_attempt", lambda _: SimpleNamespace(
        model_dump=lambda: {"attempt_id": "new"}))
    response = video.analyze_image(UploadFile(file=io.BytesIO(b"image")), "p", "s", "", False)
    assert (response["media_type"], response["student_id"], response["problem_id"]) == ("image", "s", "p")
    def rejected(_):
        raise HTTPException(409, "finished concurrently")
    monkeypatch.setattr(attempts, "analyze_attempt", rejected)
    with pytest.raises(HTTPException) as error:
        video.analyze_image(UploadFile(file=io.BytesIO(b"image")), "p", "s", "", False)
    assert error.value.status_code == 409


#: 一份**结构不合格**的复盘：`encouragement` 里写了"没有发生某事"（契约 ⑦-3 第③条判违规）。
#: 用它验证**照片路径也要过闸门** —— 原先照片路径直接把 `report.model_dump()` 返回，
#: **结构不合格也当成功给出去**（视频路径的即时复盘走 `review_video()`，是过闸门的）。
_BAD_REVIEW = {
    "timeline": "他只写了一轮。",
    "sticking_points": [], "key_concept": "", "next_step": "",
    "encouragement": "整段没有出现中途换向，主线也用对了。",
    "answer_check": None, "evidence": [], "findings": [],
    "review_schema_version": "v3+prompt-v3",
}


def test_image_review_passes_the_same_structural_gate(media, monkeypatch):
    """**照片路径不许绕过结构闸门**（2026-10-09 收口）。

    ⚠️ 但照片路径**没有重试路径**（不落盘、无 run_id、`review_available=false`），
    所以不能像视频那样抛 502 —— 那会让学生连识别原文都拿不到。
    ⇒ 按契约 §2.1 的三态：**不过闸门就不返回成功结果**（`review=None` + `failed`），
      识别原文与即时提示照常返回（不变式①：复盘失败不得击穿批改结果）。
    """
    monkeypatch.setattr(service, "call_vision", lambda *_args, **_kwargs: ('{"text":"step"}', {}))
    monkeypatch.setattr(attempts, "analyze_attempt", lambda _: SimpleNamespace(
        model_dump=lambda: {"attempt_id": "new"}))

    # ① 结构不合格 ⇒ 不返回成功结果，但批改结果照常
    monkeypatch.setattr(review_builder, "review_trajectory",
                        lambda *_: SimpleNamespace(model_dump=lambda: dict(_BAD_REVIEW)))
    bad = video.analyze_image(UploadFile(file=io.BytesIO(b"image")), "p", "s", "", True)
    assert bad["review"] is None
    assert bad["review_status"] == video.REVIEW_FAILED
    assert bad["recognized_text"] == "step", "复盘被挡不该击穿批改结果"

    # ② 结构合格 ⇒ 照常返回
    good = {**_BAD_REVIEW, "encouragement": "你写的 1×6=3v 这一点是对的。"}
    monkeypatch.setattr(review_builder, "review_trajectory",
                        lambda *_: SimpleNamespace(model_dump=lambda: dict(good)))
    ok = video.analyze_image(UploadFile(file=io.BytesIO(b"image")), "p", "s", "", True)
    assert ok["review"] is not None
    assert ok["review_status"] == video.REVIEW_READY


@pytest.mark.parametrize("strategy", ["fixed", "keyframe"])
@pytest.mark.parametrize("frame_count,deadline,expected", [
    (2, 100, "eof"), (3, 100, "max_frames"), (3, 1, "range_end")])
def test_extraction_stop_reason_uses_actual_remaining_video(
        monkeypatch, strategy, frame_count, deadline, expected):
    class Capture:
        def __init__(self):
            self.left = frame_count
        def read(self):
            self.left -= 1
            return self.left >= 0, object()
    class Sink:
        count = 0
        def full(self, *_args, **_kwargs):
            self.count += 1
            return True
        def n_full(self):
            return self.count
        def auto_rois(self, *_args):
            return 0
    monkeypatch.setattr(service, "_probe_signatures", lambda _: {})
    monkeypatch.setattr(service, "_compare", lambda *_: SimpleNamespace(total=1, local=1))
    options = service.VideoAnalysisOptions(strategy=strategy, max_frames=2, every=1,
                                           probe_every=1, min_gap_sec=0)
    extractor = service._extract_fixed if strategy == "fixed" else service._extract_keyframes
    stats = extractor(Capture(), 1, options, Sink(), deadline)
    assert stats["stop_reason"] == expected


def test_pipeline_reports_failures_without_guessing_cap(monkeypatch, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    frames = [{"type": "keyframe", "ts": 2}]
    monkeypatch.setattr(service, "extract_frames", lambda *_args, **_kwargs: (
        frames, {"stop_reason": "eof", "scanned_until_sec": 4}))
    monkeypatch.setattr(service, "build_vl_client", lambda _: (None, "fake"))
    monkeypatch.setattr(service, "analyze_frames", lambda *_args, **_kwargs: [
        {"frame": "frame.jpg", "index": 1, "timestamp_sec": 2, "error": "vision timeout"}])
    monkeypatch.setattr(service, "export_results", lambda *_args, **_kwargs: None)
    summary = service.run_pipeline(source, tmp_path / "result",
                                   service.VideoAnalysisOptions(max_frames=1, start=1, end=4))
    coverage = summary["coverage"]
    assert coverage["truncated"] is False
    assert coverage["extraction_stop_reason"] == "eof"
    assert coverage["requested_start_sec"] == 1
    assert coverage["requested_end_sec"] == 4
    assert coverage["failed_frame_count"] == 1
    assert coverage["failed_frames"][0]["error"] == "vision timeout"
