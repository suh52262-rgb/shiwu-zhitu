from types import SimpleNamespace

import pytest

from app.vision import service


class Capture:
    def __init__(self, pts):
        self.pts, self.index = pts, -1

    def read(self):
        self.index += 1
        return (True, object()) if self.index < len(self.pts) else (False, None)

    def get(self, key):
        assert key == service.cv2.CAP_PROP_POS_MSEC
        return self.pts[self.index] * 1000


class Sink:
    def __init__(self):
        self.samples = []

    def full(self, frame, ts, index, kind, score, **kwargs):
        self.samples.append((ts, index, kind))
        return True

    def auto_rois(self, *args):
        return 0

    def n_full(self):
        return len(self.samples)


def install_static_probe(monkeypatch):
    monkeypatch.setattr(service, "_probe_signatures", lambda _: {})
    monkeypatch.setattr(service, "_compare", lambda *a: SimpleNamespace(total=0., local=0.))


def test_fixed_every_uses_real_pts_instead_of_average_fps():
    cap, sink = Capture([0, .1, 10.1, 10.2]), Sink()
    stats = service._extract_fixed(cap, 30, service.VideoAnalysisOptions(every=1, max_frames=0), sink, float("inf"))
    assert [s[0] for s in sink.samples] == [0, .1, 10.1, 10.2]
    assert stats["timestamp_source"] == "decoded_pts" and stats["timestamp_reliable"]


def test_fixed_fps_samples_by_elapsed_time_on_variable_rate_input():
    sink = Sink()
    service._extract_fixed(Capture([0, .1, .2, 2, 2.1, 4]), 30,
        service.VideoAnalysisOptions(fps=1, max_frames=0), sink, float("inf"))
    assert [s[0] for s in sink.samples] == [0, 2, 4]


def test_static_representative_depends_on_pts_not_frame_count(monkeypatch):
    install_static_probe(monkeypatch)
    sink = Sink()
    stats = service._extract_keyframes(Capture([0, .1, 10.1, 10.2]), 30,
        service.VideoAnalysisOptions(probe_every=1, min_gap_sec=0, static_every_sec=5, max_frames=0),
        sink, float("inf"))
    assert sink.samples == [(0, 0, "keyframe"), (10.1, 2, "static_rep")]
    assert stats["scanned_until_sec"] == 10.2


@pytest.mark.parametrize("mode", ["fixed", "keyframe"])
def test_limit_lookahead_respects_pts_end_boundary(monkeypatch, mode):
    install_static_probe(monkeypatch)
    sink = Sink()
    opts = service.VideoAnalysisOptions(every=1, probe_every=1, min_gap_sec=0,
        static_every_sec=.1, max_frames=2)
    extract = service._extract_fixed if mode == "fixed" else service._extract_keyframes
    stats = extract(Capture([0, 1, 11]), 30, opts, sink, 10)
    assert stats["stop_reason"] == "range_end"
    assert [s[0] for s in sink.samples] == [0, 1]


def test_broken_pts_fallback_is_explicit_and_never_moves_backwards():
    cap = Capture([0, 2, float("nan"), 1, 4])
    clock = service._VideoClock(cap, 30)
    times = []
    for i in range(5):
        cap.read(); times.append(clock.timestamp(i))
    stats = {}; clock.annotate(stats)
    assert times == [0, 2, 2, 2, 4]
    assert stats["timestamp_fallback_frames"] == 2
    assert not stats["timestamp_reliable"]
    assert stats["timestamp_source"] == "mixed_pts_and_frame_index"
