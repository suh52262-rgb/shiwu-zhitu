"""Detection resize is opt-in; thin handwriting remains a known risk."""
import cv2
import numpy as np
import pytest

from app.vision.keyframe import compare, probe_signatures
from app.vision.service import VideoAnalysisOptions
from app.vision import service


def test_default_signatures_keep_original_geometry_and_input():
    image = np.full((800, 1200, 3), 255, np.uint8)
    original = image.copy()
    assert probe_signatures(image)["gray_full"].shape == (800, 1200)
    assert probe_signatures(image, max_side=600)["gray_full"].shape == (400, 600)
    np.testing.assert_array_equal(image, original)
    assert VideoAnalysisOptions().detection_max_side == 0


@pytest.mark.parametrize("event", ["write", "erase", "move"])
def test_candidate_retains_visible_change_for_broad_strokes(event):
    before = np.full((800, 1200, 3), 245, np.uint8)
    cv2.putText(before, "F=ma", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    after = before.copy()
    if event == "write":
        cv2.line(after, (700, 350), (760, 410), (0, 0, 0), 6)
    elif event == "erase":
        after[300:430, 200:500] = 245
    else:
        after = np.roll(after, 45, axis=1)
    assert compare(probe_signatures(after, max_side=600),
                   probe_signatures(before, max_side=600)).total > 0.04


def test_tiny_stroke_can_be_attenuated_so_resize_is_not_quality_equivalent():
    before = np.full((1600, 1000, 3), 255, np.uint8)
    after = before.copy()
    cv2.line(after, (500, 800), (510, 805), (0, 0, 0), 1)
    full = probe_signatures(after)
    small = probe_signatures(after, max_side=400)
    assert full["gray_full"].min() == 0
    assert small["gray_full"].min() > 0  # Subpixel ink loses contrast.
    # Final scores need not decline monotonically: phash/grid alignment may
    # also amplify small changes. Do not equate a score with OCR recall.


def test_anchor_score_reuse_preserves_distinct_anchor_comparison(monkeypatch):
    from types import SimpleNamespace
    class Capture:
        index = -1
        def read(self):
            self.index += 1
            return (True, self.index) if self.index < 4 else (False, None)
        def get(self, _):
            return self.index * 1000
    class Sink:
        def full(self, *args, **kwargs): return True
        def auto_rois(self, *args): return 0
        def n_full(self): return 0
    calls = []
    monkeypatch.setattr(service, "_probe_signatures", lambda f: {"frame": f})
    def compare(cur, prev):
        calls.append((cur["frame"], None if prev is None else prev["frame"]))
        return SimpleNamespace(total=0 if prev is None else (cur["frame"] - prev["frame"]) * .01, local=0)
    monkeypatch.setattr(service, "_compare", compare)
    stats = service._extract_keyframes(Capture(), 1,
        VideoAnalysisOptions(probe_every=1, static_every_sec=0, change_threshold=.04), Sink(), float("inf"))
    assert calls == [(0, None), (1, 0), (2, 1), (2, 0), (3, 2), (3, 0)]
    assert stats["scores"][2]["anchor"] == pytest.approx(.02)
