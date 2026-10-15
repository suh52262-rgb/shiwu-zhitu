from app.vision.teaching_snapshots import coalesce_snapshots


def test_repeated_ocr_is_suppressed_without_changing_source_or_claiming_stuck():
    source = [{"ts": 0, "text": "F = ma"}, {"ts": 20, "text": " F  = ma "},
              {"ts": 40, "text": "F = ma"}]
    chosen, skipped = coalesce_snapshots(source)
    assert chosen == [source[0]] and chosen[0] is not source[0]
    assert [r["ts"] for r in skipped] == [20, 40]
    assert all(r["duplicate_of_ts"] == 0 and r["cognitive_stuck"] is None for r in skipped)
    assert "observed_until_ts" not in source[0]


def test_sign_units_and_return_to_previous_work_are_never_dropped():
    texts = ["I=19 N", "I=-19 N", "I=-19 N·s", "I=19 N"]
    source = [{"ts": i * 20, "text": t} for i, t in enumerate(texts)]
    assert coalesce_snapshots(source) == (source, [])


def test_line_boundaries_are_preserved_and_empty_input_is_safe():
    source = [{"ts": 0, "text": "m a"}, {"ts": 20, "text": "ma"},
              {"ts": 40, "text": "m\na"}]
    assert coalesce_snapshots(source) == (source, [])
    assert coalesce_snapshots([]) == ([], [])
