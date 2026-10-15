"""Real-model review of a saved acceptance state; no new video or student evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--problem-id", default="demo_03")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be new; earlier model evidence is immutable")
    args.out.mkdir(parents=True)
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    source = args.source.resolve()
    record_data = read(source / f"{args.problem_id}_record.json")
    revision = read(source / f"{args.problem_id}_review_revision.json")
    prior = revision["fresh"]
    record_data["state"] = revision["correction"]["state"]
    from app.config import get_settings
    settings = get_settings()
    settings.llm_provider = "deepseek"
    settings.llm_model = settings.diagnosis_model = "deepseek-flash"
    from app.models.attempt import AttemptRecord
    from app.services import attempt_store
    attempt_store._STORE = attempt_store.AttemptStore(persistent=False)
    record = AttemptRecord.model_validate(record_data)
    attempt_store._STORE._memory[record.attempt_id] = record
    from app.api import video
    video.RUNS_DIR = args.out.resolve() / "perception"
    original = source / "perception" / prior["run_id"]
    target = video.RUNS_DIR / prior["run_id"]
    target.mkdir(parents=True)
    for name in ("review_input.json", "review_report.json", "review_inputs", "review_reports"):
        path = original / name
        if path.is_dir():
            shutil.copytree(path, target / name)
        elif path.exists():
            shutil.copy2(path, target / name)
    files = {}
    for base in (ROOT / "backend" / "app", ROOT / "prompts"):
        for path in base.rglob("*"):
            if path.suffix in {".py", ".md"}:
                relative = path.relative_to(ROOT)
                data = path.read_bytes()
                files[str(relative)] = hashlib.sha256(data).hexdigest()
                frozen = args.out / "source_snapshot" / relative
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(data)
    save(args.out / "manifest.json", {"scope": "Recorded corrected server state replay with real model; no new video processing, no DB or new student input", "source": str(source), "model": "deepseek-flash", "files": files})
    from fastapi.testclient import TestClient
    from app.main import app
    body = {key: prior[key] for key in ("student_id", "problem_id", "attempt_id", "run_id", "input_revision")}
    with TestClient(app) as client:
        start = time.perf_counter()
        response = client.post("/video/review", json=body)
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        repeat = client.post("/video/review", json=body)
        save(args.out / "review.json", {"http_status": response.status_code, "elapsed_s": elapsed,
            "response": response.json(), "repeat_status": repeat.status_code,
            "repeat_elapsed_s": time.perf_counter() - start, "repeat_response": repeat.json(),
            "previous_generation_fingerprint": prior["generation_fingerprint"]})
        assert response.status_code == repeat.status_code == 200
        assert response.json()["cached"] is False and repeat.json()["cached"] is True
        assert response.json()["generation_fingerprint"] != prior["generation_fingerprint"]
        assert response.json()["review_input_fingerprint"] == prior["review_input_fingerprint"]
    print(f"COMPLETE real-model recorded-state review {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
