"""Export 1–2 second visual observations without OCR or teaching-model calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.vision.observation import observe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.)
    parser.add_argument("--roi", type=float, nargs=4, default=[0., 0., 1., 1.],
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    result = observe_video(args.video, args.out, interval=args.interval, roi=args.roi)
    print(f"samples={result['sample_count']} candidates={len(result['candidates'])} elapsed={result['elapsed_sec']}s")


if __name__ == "__main__":
    main()
