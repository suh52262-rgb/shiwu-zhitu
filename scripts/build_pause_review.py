"""Propose low-visual-change intervals for review, never cognitive stuck labels.

Uses exported keyframes, a fixed handwriting crop and cumulative ink-mask change.
Missing/widely separated frames break evidence; blank starts are excluded.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import cv2
import numpy as np


def changed_pixels(left, right):
    """Ignore one-pixel boundary jitter introduced by video/JPEG compression."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    ld = cv2.dilate(left.astype(np.uint8), kernel).astype(bool)
    rd = cv2.dilate(right.astype(np.uint8), kernel).astype(bool)
    return np.count_nonzero(left & ~rd) + np.count_nonzero(right & ~ld)


def intervals(samples, *, min_seconds=5, max_gap=6, min_pixels=25):
    """samples: (timestamp, binary handwriting mask, path). Thresholds uncalibrated."""
    out, anchor, start, last, previous, first_path, last_path = [], None, None, None, None, None, None
    def emit(end, end_path):
        if start is not None and last is not None and last - start >= min_seconds:
            out.append({"start_sec": round(start, 3), "end_sec": round(last, 3),
                        "observed_span_sec": round(last - start, 3),
                        "next_change_or_boundary_sec": round(end, 3),
                        "start_frame": str(first_path), "end_frame": str(end_path),
                        "last_stable_frame": str(last_path),
                        "label": "low_visual_change_candidate", "review_status": "pending",
                        "cognitive_stuck": None})
    for ts, mask, path in samples:
        if anchor is None or (previous is not None and ts - previous > max_gap):
            if previous is not None:
                emit(previous, last_path)
            anchor, start, last, first_path = mask.copy(), ts, ts, path
        elif changed_pixels(mask, anchor) >= min_pixels:
            emit(ts, path)
            anchor, start, last, first_path = mask.copy(), ts, ts, path
        else:
            last = ts
        if np.count_nonzero(anchor) < min_pixels:
            start = None  # blank before beginning is not a solving pause
        previous, last_path = ts, path
    if previous is not None:
        emit(previous, last_path)
    return out


def samples_from(directory):
    paths = []
    for p in directory.glob("*s.jpg"):
        if not p.name.startswith(("kf_", "sc_", "st_")):
            continue  # ROI crops cannot be compared to full-canvas frames
        match = re.search(r"_t([0-9.]+)s", p.name)
        if match:
            paths.append((float(match[1]), p))
    for ts, p in sorted(paths):
        frame = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        gray = cv2.resize(frame, (480, 768))
        # This crop is specific to the authorized portrait notebook recordings.
        canvas = gray[62:730, 35:470]
        mask = cv2.GaussianBlur(canvas, (3, 3), 0) < 150
        yield ts, mask, p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    samples = list(samples_from(args.frames_dir))
    candidates = intervals(samples)
    payload = {"source": str(args.frames_dir), "sample_count": len(samples),
               "thresholds": {"minimum_span_sec": 5, "maximum_sample_gap_sec": 6,
                              "cumulative_changed_pixels": 25, "ink_gray_below": 150,
                              "boundary_jitter_tolerance_pixels": 1},
               "scope": "Uncalibrated visual-change screening; not attention, thinking or stuck detection. Sampled frames do not prove continuous inactivity.",
               "candidates": candidates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["start_sec", "end_sec", "observed_span_sec", "next_change_or_boundary_sec", "review_status", "cognitive_stuck", "reviewer", "notes"]
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(candidates)
    print(json.dumps({"samples": len(samples), "candidates": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
