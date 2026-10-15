import numpy as np

from scripts.build_pause_review import intervals


def mask(*positions):
    a = np.zeros((5, 5), dtype=bool)
    for p in positions:
        a.flat[p] = True
    return a


def test_blank_start_and_large_missing_gap_are_not_solving_pause():
    samples = [(0, mask(), "0"), (3, mask(), "3"), (6, mask(), "6"),
               (7, mask(1), "7"), (20, mask(1), "20")]
    assert intervals(samples, min_pixels=1) == []


def test_cumulative_small_writing_resets_pause_and_preserves_provenance():
    samples = [(0, mask(1, 2), "a"), (2, mask(1, 2), "b"), (4, mask(1, 2), "c"),
               (6, mask(1, 2), "d"), (8, mask(1, 2, 20), "e"), (10, mask(1, 2, 20, 24), "f")]
    result = intervals(samples, min_pixels=2)
    assert len(result) == 1
    assert result[0]["observed_span_sec"] == 8
    assert result[0]["next_change_or_boundary_sec"] == 10
    assert result[0]["cognitive_stuck"] is None
    assert result[0]["review_status"] == "pending"
    assert result[0]["start_frame"] == "a" and result[0]["end_frame"] == "f"
