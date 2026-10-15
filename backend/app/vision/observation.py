"""Low-cost video observations, separate from OCR and teaching decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from app.vision.service import _VideoClock


def ink_mask(frame, roi, width=480):
    """Fractional ROI is explicit; no automatic claim of a handwriting region."""
    if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("invalid fractional observation ROI")
    if width < 1:
        raise ValueError("observation mask width must be positive")
    h, w = frame.shape[:2]
    left, top, right, bottom = roi
    canvas = frame[int(top * h):int(bottom * h), int(left * w):int(right * w)]
    if not canvas.size:
        raise ValueError("observation ROI is empty after pixel rounding")
    scaled = cv2.resize(canvas, (width, max(1, round(canvas.shape[0] * width / canvas.shape[1]))),
                        interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0) < 150, scaled


def ink_delta(before, after):
    kernel = np.ones((3, 3), np.uint8)
    old = cv2.dilate(before.astype(np.uint8), kernel).astype(bool)
    new = cv2.dilate(after.astype(np.uint8), kernel).astype(bool)
    return int(np.count_nonzero(after & ~old)), int(np.count_nonzero(before & ~new))


def observe_samples(samples, *, minimum_span=3., maximum_gap=2.5, changed_pixels=25):
    """Pairwise records plus cumulative-anchor candidates; sampled inactivity only.

    samples contain decoded PTS, ink mask, evidence path. Gaps and blank canvas
    cannot contribute to a solving-pause candidate. Subthreshold strokes
    accumulate against the anchor, avoiding endless 'no change' classification.
    """
    observations, candidates = [], []
    previous, anchor, stable_start, stable_last = None, None, None, None
    def flush():
        if stable_start and stable_last and stable_last[0] - stable_start[0] >= minimum_span:
            candidates.append({"start_sec": stable_start[0], "end_sec": stable_last[0],
                "sampled_span_sec": round(stable_last[0] - stable_start[0], 3),
                "start_evidence": str(stable_start[2]), "end_evidence": str(stable_last[2]),
                "label": "low_visual_change_candidate", "review_status": "pending",
                "cognitive_stuck": None})
    for sample in samples:
        ts, mask, evidence = sample
        if previous is None:
            kind, added, removed = "initial", 0, 0
        elif ts <= previous[0] or ts - previous[0] > maximum_gap:
            kind, added, removed = "evidence_gap", 0, 0
        else:
            added, removed = ink_delta(previous[1], mask)
            kind = "visual_change" if added + removed >= changed_pixels else "low_visual_change"
        observations.append({"timestamp_sec": ts, "kind": kind, "added_ink_pixels": added,
            "removed_ink_pixels": removed, "ink_pixels": int(mask.sum()), "evidence": str(evidence)})
        blank = int(mask.sum()) < changed_pixels
        if anchor is None or kind in {"initial", "evidence_gap"} or blank:
            flush()
            anchor = mask.copy()
            stable_start = stable_last = None if blank else sample
        else:
            a, r = ink_delta(anchor, mask)
            if a + r >= changed_pixels:
                flush()
                anchor = mask.copy()
                stable_start = stable_last = sample
            else:
                if stable_start is None:
                    stable_start = sample
                stable_last = sample
        previous = sample
    flush()
    return observations, candidates


def observe_video(source: Path, out: Path, *, interval=1., roi=(0., 0., 1., 1.)):
    """Sequential decode only; no model, DB writes, or teaching-state mutations."""
    if not .5 <= interval <= 2.:
        raise ValueError("observation interval must be 0.5–2 seconds")
    if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("invalid fractional observation ROI")
    if out.exists():
        raise ValueError("output must be new to preserve evidence")
    started = time.perf_counter()
    out.mkdir(parents=True)
    frames = out / "frames"; frames.mkdir()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ValueError("video cannot be opened")
    clock = _VideoClock(cap, cap.get(cv2.CAP_PROP_FPS) or 25.)
    sampled, index, last = [], 0, -float("inf")
    previous_decoded, decoded_pts, presentation_holds = None, [], []
    def save_preview(frame, name):
        _, preview = ink_mask(frame, roi)
        evidence = frames / name
        success, encoded = cv2.imencode(".jpg", preview)
        if not success:
            raise ValueError("observation evidence encoding failed")
        evidence.write_bytes(encoded.tobytes())
        return str(evidence.resolve())
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = clock.timestamp(index)
            decoded_pts.append(ts)
            if previous_decoded is not None:
                previous_frame, previous_ts, previous_index = previous_decoded
                if ts - previous_ts >= 3.:
                    # Consecutive decoded presentation times, not skipped
                    # observation samples. Never fabricate intermediate frames.
                    presentation_holds.append({
                        "label": "encoded_presentation_hold", "start_sec": round(previous_ts, 6),
                        "end_sec_exclusive": round(ts, 6), "duration_sec": round(ts - previous_ts, 6),
                        "previous_decoded_index": previous_index, "next_decoded_index": index,
                        "start_evidence": save_preview(previous_frame, f"hold_{previous_index:06d}_before.jpg"),
                        "next_frame_evidence": save_preview(frame, f"hold_{index:06d}_after.jpg"),
                        "cognitive_stuck": None, "review_status": "pending",
                        "meaning": "Previous frame presentation held until next decoded PTS; not proof of recording inactivity or cognitive stuck."})
            if ts - last >= interval:
                mask, preview = ink_mask(frame, roi)
                evidence = frames / f"obs_{index:06d}_t{ts:010.3f}s.jpg"
                success, encoded = cv2.imencode(".jpg", preview)
                if not success:
                    raise ValueError("observation evidence encoding failed")
                evidence.write_bytes(encoded.tobytes())
                sampled.append((round(ts, 3), mask, str(evidence.resolve())))
                last = ts
            previous_decoded = (frame, ts, index)
            index += 1
    finally:
        cap.release()
    observations, candidates = observe_samples(sampled, maximum_gap=interval * 2.5)
    timing = {}; clock.annotate(timing)
    if not timing["timestamp_reliable"]:
        candidates = []  # Approximate FPS fallback cannot support precise short spans.
        presentation_holds = []
    gaps = np.diff(decoded_pts)
    result = {"source": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "interval_sec": interval, "roi": list(roi), "timestamp": timing,
        "decoded_frames": index, "sample_count": len(sampled),
        "decoded_pts_sec": decoded_pts,
        "presentation_timing": {"gap_count": len(gaps),
            "max_gap_sec": float(gaps.max()) if len(gaps) else None,
            "gaps_ge_3_sec": int(np.count_nonzero(gaps >= 3.)),
            "gap_percentiles_sec": {str(p): float(np.percentile(gaps, p)) for p in (50, 90, 99)} if len(gaps) else {},
            "interpretation": "Consecutive decoder PTS; holds assume standard presentation semantics. Last-frame duration unknown; no tail extrapolation."},
        "presentation_holds": presentation_holds,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "scope": "Uncalibrated sampled visual observations; no cognitive stuck or intervention decision.",
        "limitations": ["480-pixel ROI resize, blur and ink threshold can erase thin or faint strokes.",
                        "One-pixel spatial tolerance can ignore adjacent writing or partial erasure.",
                        "Activity between samples is unobserved; low visual change is not proof of no writing."],
        "observations": observations, "candidates": candidates}
    (out / "observations.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
