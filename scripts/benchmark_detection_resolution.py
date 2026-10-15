"""Offline paired extraction; no models, DB, or random video seeks.

All probe scores and retained timestamps are saved, including threshold
crossings. Baseline detections are a comparator, never human ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.vision.service import VideoAnalysisOptions, extract_frames


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def contact(out, runs, pairs):
    """Compare retained source views; not a claim of exhaustive event recall."""
    choices = [p for p in pairs if p["threshold_missed"] and p["baseline"]["keep"]][:8]
    if not choices:
        choices = [p for p in pairs if p["selection_changed"]][:8]
    if not choices:
        types = {r["index"]: r["type"] for r in runs["candidate"]["records"] if r["type"] != "roi"}
        choices = [{"ts": r["ts"]} for r in runs["baseline"]["records"]
                   if r["type"] != "roi" and types.get(r["index"]) != r["type"]][:8]
    if not choices:
        return
    page = Image.new("RGB", (900, 450 * len(choices)), "white")
    draw = ImageDraw.Draw(page)
    for row, choice in enumerate(choices):
        timestamp = choice["ts"]
        for col, label in enumerate(("baseline", "candidate")):
            frames = [r for r in runs[label]["records"] if r["type"] != "roi"]
            record = min(frames, key=lambda r: abs(r["ts"] - timestamp))
            image = Image.open(out / label / "frames" / record["file"])
            image.thumbnail((440, 415))
            page.paste(image, (col * 450, row * 450 + 30))
            draw.text((col * 450 + 5, row * 450 + 5),
                      f"{label}: probe={timestamp:.3f}, retained={record['ts']:.3f}", fill="black")
    page.save(out / "selection_difference_contact.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()
    manifest = {"files": {}, "code": {}, "runs": {}}
    for rel in ("backend/app/vision/keyframe.py", "backend/app/vision/service.py", __file__):
        path = Path(rel)
        manifest["code"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for number in ("02", "03"):
        video = args.video_dir / f"真实例子作答{number}.mp4"
        manifest["files"][video.name] = hashlib.sha256(video.read_bytes()).hexdigest()
        runs = {}
        for label, side in (("baseline", 0), ("candidate", args.max_side)):
            options = VideoAnalysisOptions(probe_every=8, detection_max_side=side, dry_run=True)
            print(f"START {number} {label}", flush=True)
            records, stats = extract_frames(video, args.out / number / label / "frames", options)
            # JPEG bytes are already written by FrameSink. Keep metadata only
            # so the evidence file is both JSON-safe and reasonably small.
            metadata = [{key: value for key, value in record.items()
                         if not isinstance(value, (bytes, bytearray))} for record in records]
            result = {"options": options.model_dump(), "records": metadata, "stats": stats}
            save(args.out / number / f"{label}.json", result)
            runs[label] = result
            print(f"DONE {number} {label}: {stats['kept']} frames {stats['timings']}", flush=True)
        pairs = []
        for a, b in zip(runs["baseline"]["stats"]["scores"], runs["candidate"]["stats"]["scores"]):
            assert a["ts"] == b["ts"], "PTS or probe sequence changed"
            pairs.append({"ts": a["ts"], "baseline": a, "candidate": b,
                          "threshold_missed": a["score"] >= a.get("thr", .04) and b["score"] < b.get("thr", .04),
                          "selection_changed": a["keep"] != b["keep"]})
        save(args.out / number / "paired_probes.json", pairs)
        contact(args.out / number, runs, pairs)
        summary = {label: {"kept": value["stats"]["kept"], "timings": value["stats"]["timings"],
                           "scanned_until_sec": value["stats"]["scanned_until_sec"]}
                   for label, value in runs.items()}
        summary["comparison"] = {"probes": len(pairs), "threshold_missed": sum(p["threshold_missed"] for p in pairs),
                                 "selection_changed": sum(p["selection_changed"] for p in pairs),
                                 "quality_ground_truth": False}
        manifest["runs"][number] = summary
        save(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
