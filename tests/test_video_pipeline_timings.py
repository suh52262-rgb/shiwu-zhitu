"""Wall-clock phase accounting: concurrency/retries remain inside vision time."""
from types import SimpleNamespace

import pytest

from app.vision import service


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


@pytest.mark.parametrize("strategy", ["fixed", "keyframe"])
def test_extraction_breakdown_is_disjoint_and_includes_eof(monkeypatch, tmp_path, strategy):
    clock = Clock()
    monkeypatch.setattr(service.time, "perf_counter", clock)

    class Capture:
        index = -1

        def isOpened(self):
            return True

        def get(self, key):
            values = {service.cv2.CAP_PROP_FPS: 1, service.cv2.CAP_PROP_FRAME_COUNT: 3,
                      service.cv2.CAP_PROP_FRAME_WIDTH: 32, service.cv2.CAP_PROP_FRAME_HEIGHT: 32,
                      service.cv2.CAP_PROP_POS_MSEC: self.index * 1000}
            return values[key]

        def read(self):
            clock.advance(1)
            self.index += 1
            return (True, object()) if self.index < 3 else (False, None)

        def release(self):
            pass

    class Sink:
        def __init__(self, *args):
            self.records = []

        def full(self, frame, ts, *args, **kwargs):
            clock.advance(5)
            self.records.append({"type": "keyframe", "ts": ts})
            return True

        def auto_rois(self, *args):
            clock.advance(7)
            return 0

        def n_full(self):
            return len(self.records)

    def signature(frame):
        clock.advance(2)
        return {}

    def compare(*args):
        clock.advance(3)
        return SimpleNamespace(total=0, local=0)

    monkeypatch.setattr(service.cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(service, "_FrameSink", Sink)
    monkeypatch.setattr(service, "_probe_signatures", signature)
    monkeypatch.setattr(service, "_compare", compare)
    frames, stats = service.extract_frames(tmp_path / "video.mp4", tmp_path / "frames",
        service.VideoAnalysisOptions(strategy=strategy, every=1, probe_every=1,
                                     static_every_sec=1, min_gap_sec=0, max_frames=0))
    timing = stats["timings"]
    assert len(frames) == 3
    assert timing["decode_read_sec"] == 4  # Three frames plus final EOF read.
    assert timing["frame_output_sec"] == 36  # ROI work included once.
    # Each retained probe becomes both the previous and anchor reference;
    # compare it once, reusing the mathematically identical anchor score.
    assert timing["change_detection_sec"] == (15 if strategy == "keyframe" else 0)
    assert timing["extraction_total_sec"] == sum(timing[k] for k in (
        "decode_read_sec", "change_detection_sec", "frame_output_sec", "extraction_other_sec"))


@pytest.mark.parametrize("dry_run", [False, True])
def test_pipeline_reports_wall_time_without_double_counting_extraction(monkeypatch, tmp_path, dry_run):
    clock = Clock()
    monkeypatch.setattr(service.time, "perf_counter", clock)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"stub")

    def extract(*args, **kwargs):
        clock.advance(8)
        return [{"type": "keyframe", "ts": 0}], {"timings": {
            "decode_read_sec": 5, "change_detection_sec": 1, "frame_output_sec": 2,
            "extraction_total_sec": 8, "extraction_other_sec": 0}}

    def client(*args):
        clock.advance(2)
        return object(), "test-model"

    def analyze(*args, **kwargs):
        clock.advance(20)  # Concurrent worker requests measured as one wall interval.
        return [{"model_used": "test-model"}]

    def export(*args, **kwargs):
        clock.advance(3)

    monkeypatch.setattr(service, "extract_frames", extract)
    monkeypatch.setattr(service, "build_vl_client", client)
    monkeypatch.setattr(service, "analyze_frames", analyze)
    monkeypatch.setattr(service, "export_results", export)
    monkeypatch.setattr(service.cv2, "VideoCapture", lambda _: SimpleNamespace(
        isOpened=lambda: False, release=lambda: None))
    summary = service.run_pipeline(video, tmp_path / "out", service.VideoAnalysisOptions(dry_run=dry_run))
    timing = summary["timings"]
    assert timing["extraction_sec"] == 8
    assert timing["vision_sec"] == (0 if dry_run else 20)
    assert timing["client_setup_sec"] == (0 if dry_run else 2)
    assert timing["export_sec"] == (0 if dry_run else 3)
    assert timing["total_sec"] == (8 if dry_run else 33)
    assert timing["total_sec"] == sum(timing[k] for k in (
        "extraction_sec", "vision_sec", "client_setup_sec", "export_sec", "other_sec"))
    assert timing["extraction"]["decode_read_sec"] == 5
