import numpy as np
import cv2
import pytest

from app.vision.observation import observe_samples, ink_mask


def mask(*positions):
    arr = np.zeros((20, 20), dtype=bool)
    for pos in positions:
        arr[pos] = True
    return arr


def test_blank_start_and_missing_interval_do_not_count_as_pause():
    blank = mask()
    ink = mask((2, 2), (3, 2))
    records, candidates = observe_samples([(0, blank, "a"), (1, blank, "b"),
        (2, blank, "c"), (3, ink, "d"), (20, ink, "e")], changed_pixels=2)
    assert not candidates
    assert records[-1]["kind"] == "evidence_gap"


def test_stable_samples_have_evidence_but_never_cognitive_label():
    ink = mask((2, 2), (3, 2))
    records, candidates = observe_samples([(i, ink, str(i)) for i in range(5)], changed_pixels=2)
    assert candidates[0]["sampled_span_sec"] == 4
    assert candidates[0]["cognitive_stuck"] is None
    assert candidates[0]["start_evidence"] == "0" and candidates[0]["end_evidence"] == "4"


def test_small_strokes_accumulate_and_erasure_is_observed():
    samples = [(0, mask((2, 2), (3, 2)), "a"),
               (1, mask((2, 2), (3, 2), (8, 8)), "b"),
               (2, mask((2, 2), (3, 2), (8, 8), (14, 14)), "c"),
               (3, mask((2, 2), (3, 2)), "d")]
    records, candidates = observe_samples(samples, changed_pixels=2, minimum_span=2)
    assert not candidates
    assert records[-1]["kind"] == "visual_change"
    assert records[-1]["removed_ink_pixels"] == 2


def test_thin_high_resolution_stroke_can_disappear_from_observation_mask():
    blank = np.full((2944, 1840, 3), 255, np.uint8)
    writing = blank.copy()
    cv2.line(writing, (900, 1000), (930, 1000), (0, 0, 0), 1)
    original = cv2.cvtColor(writing, cv2.COLOR_BGR2GRAY)
    sampled, _ = ink_mask(writing, (0, 0, 1, 1))
    assert np.count_nonzero(original < 150) > 0
    assert sampled.sum() == 0  # Known detection limit, not absence of writing.


def test_real_pixel_erasure_and_nonmonotonic_pts_are_separate_events():
    frame = np.full((400, 480, 3), 255, np.uint8)
    cv2.line(frame, (100, 100), (200, 100), (0, 0, 0), 8)
    ink, _ = ink_mask(frame, (0, 0, 1, 1))
    erased, _ = ink_mask(np.full_like(frame, 255), (0, 0, 1, 1))
    records, candidates = observe_samples([(0, ink, "a"), (1, erased, "b"),
                                          (.5, ink, "c"), (1.5, ink, "d")])
    assert records[1]["removed_ink_pixels"] > 25
    assert records[2]["kind"] == "evidence_gap"
    assert not candidates


def test_fractional_roi_that_rounds_to_no_pixels_has_clear_error():
    with pytest.raises(ValueError, match="empty after pixel rounding"):
        ink_mask(np.full((10, 10, 3), 255, np.uint8), (.1, .1, .11, .11))


def run_decoded_fixture(monkeypatch, tmp_path, pts):
    from app.vision import observation
    class Capture:
        index = -1
        def isOpened(self): return True
        def read(self):
            self.index += 1
            if self.index >= len(pts): return False, None
            frame = np.full((80, 80, 3), 255, np.uint8)
            cv2.line(frame, (20, 20), (40 + self.index, 20), (0, 0, 0), 4)
            return True, frame
        def get(self, key):
            if key == cv2.CAP_PROP_FPS: return 10
            assert key == cv2.CAP_PROP_POS_MSEC
            return pts[self.index] * 1000
        def release(self): pass
    monkeypatch.setattr(observation.cv2, "VideoCapture", lambda _: Capture())
    source = tmp_path / "test.mp4"; source.write_bytes(b"synthetic decoder fixture")
    return observation.observe_video(source, tmp_path / "out")


def test_consecutive_pts_hold_separate_from_observation_gap(monkeypatch, tmp_path):
    from pathlib import Path
    result = run_decoded_fixture(monkeypatch, tmp_path, [0, .2, 5.2, 5.4])
    assert result["sample_count"] == 2  # No invented 1,2,3,4-second samples.
    assert result["observations"][1]["kind"] == "evidence_gap"
    hold, = result["presentation_holds"]
    assert hold["start_sec"] == .2 and hold["end_sec_exclusive"] == 5.2
    assert hold["duration_sec"] == 5 and hold["cognitive_stuck"] is None
    assert hold["previous_decoded_index"] == 1 and hold["next_decoded_index"] == 2
    assert Path(hold["start_evidence"]).is_file() and Path(hold["next_frame_evidence"]).is_file()
    assert not result["candidates"]
    assert result["decoded_pts_sec"] == [0, .2, 5.2, 5.4]


def test_fallback_clock_suppresses_presentation_hold_claim(monkeypatch, tmp_path):
    result = run_decoded_fixture(monkeypatch, tmp_path, [0, .2, 5.2, -1])
    assert not result["timestamp"]["timestamp_reliable"]
    assert not result["presentation_holds"] and not result["candidates"]


def test_no_last_frame_hold_extrapolation_or_short_gap_hold(monkeypatch, tmp_path):
    result = run_decoded_fixture(monkeypatch, tmp_path, [0, 1., 2.])
    assert not result["presentation_holds"]
    assert result["presentation_timing"]["gaps_ge_3_sec"] == 0
