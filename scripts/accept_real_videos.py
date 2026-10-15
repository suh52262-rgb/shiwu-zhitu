"""Authorized real-media acceptance against disposable PostgreSQL/Qdrant services.

Outputs contain handwriting and model responses; artifacts/ is intentionally ignored.
Runs the real API in-process. RAG corpus is empty; this does not assess RAG quality.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


@contextmanager
def disposable_services():
    ids = []
    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True, encoding="utf-8").strip()
    try:
        tag = uuid4().hex[:12]
        for kind, image, port, extra in [
            ("pg", "postgres:16-alpine", "5432", ["-e", "POSTGRES_USER=shiwu_ci", "-e",
             "POSTGRES_PASSWORD=disposable_ci_password", "-e", "POSTGRES_DB=shiwu_ci"]),
            ("qd", "qdrant/qdrant:v1.12.4", "6333", [])]:
            cid = docker("run", "-d", "--name", f"shiwu-media-{kind}-{tag}", "-p", f"127.0.0.1::{port}", *extra, image)
            if not re.fullmatch(r"[0-9a-f]{64}", cid):
                raise RuntimeError("Invalid new container ID")
            ids.append(cid)
        pgport = docker("port", ids[0], "5432/tcp").split(":")[-1]
        qdport = docker("port", ids[1], "6333/tcp").split(":")[-1]
        for _ in range(30):
            ready = subprocess.run(["docker", "exec", ids[0], "pg_isready", "-U", "shiwu_ci"],
                                   capture_output=True, timeout=5).returncode == 0
            try:
                with urlopen(f"http://127.0.0.1:{qdport}/readyz", timeout=2) as r:
                    ready = ready and r.status == 200
            except OSError:
                ready = False
            if ready:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Temporary services not ready")
        # ⚠️ **`pg_isready` 通过 ≠ 可以连了。**
        #
        # postgres 官方镜像的入口点是：先起一个**临时服务**跑初始化脚本，
        # 初始化完**停掉它、再起正式服务**。`pg_isready` 完全可能在临时服务上返回 OK ——
        # 于是连接被放行，随后正式服务还没起来（或临时服务正在关闭），
        # 拿到的是 `server closed the connection unexpectedly`。
        #
        # 实测（2026-10-09）：同一份代码、同一台机器，一次跑到一半整批挂在
        # `psycopg.OperationalError: server closed the connection unexpectedly`，
        # 重跑同样的命令却一路通过 ⇒ **间歇竞态，不是代码错**。
        # 而 72 次批处理一旦这样挂掉就**整批作废**（设计上不自动续跑中断的付费请求），
        # ⇒ 代价远大于多等这几秒。
        #
        # ⇒ 就绪判据改成**真跑一次查询、且连续两次成功**（中间隔 1 秒），
        #    这样"临时服务那一瞬间"过不了关。
        stable = 0
        for _ in range(30):
            ok = subprocess.run(
                ["docker", "exec", ids[0], "psql", "-U", "shiwu_ci", "-d", "shiwu_ci",
                 "-tAc", "select 1"],
                capture_output=True, timeout=10).returncode == 0
            stable = stable + 1 if ok else 0
            if stable >= 2:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Temporary PostgreSQL never answered a real query stably")
        os.environ.update(POSTGRES_HOST="127.0.0.1", POSTGRES_PORT=pgport, POSTGRES_USER="shiwu_ci",
                          POSTGRES_PASSWORD="disposable_ci_password", POSTGRES_DB="shiwu_ci",
                          QDRANT_URL=f"http://127.0.0.1:{qdport}", PYTHONPATH=str(ROOT / "backend"),
                          CI_ISOLATED_SERVICES="1")
        yield
    finally:
        for cid in reversed(ids):
            subprocess.run(["docker", "rm", "-f", "-v", cid], check=False, capture_output=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--photo-corrected-text", default="", help="Optional explicit OCR edit to exercise correction; never a gold label")
    parser.add_argument("--generate-review", action="store_true", help="Also time the separate review request and verify cache reuse")
    parser.add_argument("--skip-photo", action="store_true", help="Limit this run to the two video paths")
    parser.add_argument("--background", action="store_true", help="Submit and poll a durable background job")
    parser.add_argument("--verify-review-revision", action="store_true", help="Exercise a semantic-preserving AI OCR edit and new real-model review; not a gold transcription")
    parser.add_argument("--videos", nargs="+", choices=["02", "03"], default=["02", "03"], help="Select authorized videos for a focused rerun")
    args = parser.parse_args()
    # Explicit acceptance profile, independent of the user's live backend.
    # Default remains v3 for old callers; an explicitly selected frozen profile
    # must not silently be replaced during media acceptance.
    os.environ.setdefault('PROMPT_REVIEW_VERSION', 'v3')
    os.environ['SHIWU_RUNTIME_PROFILE'] = 'demo'
    if args.verify_review_revision and not args.generate_review:
        parser.error("--verify-review-revision requires --generate-review")
    out = args.out.resolve()
    if out.exists():
        parser.error("Output must be a new directory, preserving previous evidence")
    out.mkdir(parents=True)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
        "authorization": "User allowed videos 02/03; frequent pauses usually <=10s or around 10-19s; exact times unannotated",
        "model": "deepseek-flash", "environment": "disposable databases; empty RAG corpus; real models and API; no browser acceptance",
        "separate_review_requested": args.generate_review,
        "background_requested": args.background,
        "review_revision_requested": args.verify_review_revision,
        "photo_edit_provenance": "explicit caller input, not teacher gold",
        "files": {}, "videos": []}
    for base in [ROOT / "backend" / "app", ROOT / "prompts"]:
        for p in sorted(base.rglob("*")):
            if p.suffix in {".py", ".md"}:
                manifest["files"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
                frozen = out / "source_snapshot" / p.relative_to(ROOT)
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(p.read_bytes())
    manifest["files"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sources = []
    for n in args.videos:
        p = args.video_dir / f"真实例子作答{n}.mp4"
        sources.append(p)
        manifest["videos"].append({"problem_id": "demo_" + n, "path": str(p), "bytes": p.stat().st_size,
                                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    write(out / "manifest.json", manifest)
    with disposable_services():
        from app.config import get_settings
        settings = get_settings()
        settings.llm_provider = "deepseek"
        settings.llm_model = "deepseek-flash"
        for role in ["step_matcher", "analyze_step", "diagnosis", "tutor"]:
            setattr(settings, role + "_model", "deepseek-flash")
        from app.db.postgres import get_engine
        from app.db.tables import Base
        Base.metadata.create_all(get_engine())
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api import video
        from app.vision import service
        from app.services.llm.tracking import track_usage
        video.RUNS_DIR = out / "perception"
        video.UPLOAD_DIR = out / "uploads"
        if args.background:
            from app.services import video_jobs
            jobs_root = (video.PROJECT_ROOT / "data" / "video_jobs").resolve()
            # Use a queue owned only by this acceptance run, preserving business jobs.
            video_jobs._queues[str(jobs_root)] = video_jobs.VideoJobQueue(out / "jobs")
        original_pipeline, original_vision = service.run_pipeline, service.call_vision
        vision_calls, pipeline_runs = [], []
        lock = threading.Lock()
        def counted_vision(*a, **kw):
            try:
                result, meta = original_vision(*a, **kw)
            except Exception as exc:
                with lock:
                    vision_calls.append({"error": type(exc).__name__})
                raise
            with lock:
                vision_calls.append(meta)
                write(out / "vision_usage.json", vision_calls)
            return result, meta
        def pipeline(path, target, options, **kw):
            options.model = "deepseek-flash"
            options.save_raw = True
            options.export_change_scores = True
            callback = kw.get("progress_cb")
            def progress(phase, message, **info):
                if callback is not None:
                    callback(phase, message, **info)
                if phase != "analyzing" or info.get("analyze_done", 0) % 25 == 0:
                    print(f"[{phase}] {message}", flush=True)
            kw["progress_cb"] = progress
            summary = original_pipeline(path, target, options, **kw)
            pipeline_runs.append({"out_dir": str(target), "options": options.model_dump(), "summary": summary})
            write(out / "pipeline_runs.json", pipeline_runs)
            return summary
        service.call_vision, service.run_pipeline = counted_vision, pipeline
        checks = []
        try:
            with TestClient(app) as client:
                write(out / 'runtime_configuration.json', client.get('/runtime/configuration').json())
                for source, item in zip(sources, manifest["videos"]):
                    problem = item["problem_id"]
                    print("START " + problem, flush=True)
                    started = time.perf_counter()
                    request_id = str(uuid4())
                    with source.open("rb") as stream:
                        response = client.post("/video/jobs" if args.background else "/video/analyze", data={"student_id": "media_accept_" + problem,
                            "problem_id": problem, "every_sec": "20", "client_request_id": request_id}, files={"file": (source.name, stream, "video/mp4")})
                    payload = response.json()
                    if args.background:
                        history = [{"elapsed_s": time.perf_counter() - started, "response": payload}]
                        write(out / (problem + "_job_submission.json"), {"http_status": response.status_code, **history[0]})
                        if response.status_code not in (200, 202):
                            raise RuntimeError("Background submission failed")
                        recovered = client.get('/video/jobs/by-request/' + request_id, params={
                            'student_id': 'media_accept_' + problem, 'problem_id': problem})
                        write(out / (problem + '_recovered_job.json'), {'http_status': recovered.status_code,
                            'client_request_id': request_id, 'response': recovered.json()})
                        if recovered.status_code != 200 or recovered.json()['job_id'] != payload['job_id']:
                            raise RuntimeError('Request-ID recovery did not return the submitted job')
                        deadline = time.monotonic() + 1800
                        previous = None
                        while True:
                            polled = client.get("/video/jobs/" + payload["job_id"], params={
                                "student_id": "media_accept_" + problem, "problem_id": problem})
                            state = polled.json()
                            if polled.status_code != 200:
                                raise RuntimeError("Background poll failed")
                            signature = (state["status"], state["phase"], json.dumps(state["progress"], sort_keys=True))
                            if signature != previous:
                                history.append({"elapsed_s": time.perf_counter() - started, "response": state})
                                write(out / (problem + "_job_history.json"), history)
                                print("JOB " + problem + " " + str(signature), flush=True)
                                previous = signature
                            if state["status"] in {"succeeded", "failed", "interrupted"}:
                                if state["status"] != "succeeded" or not state["upload_removed"]:
                                    raise RuntimeError("Background job did not finish cleanly")
                                payload = state["result"]
                                break
                            if time.monotonic() > deadline:
                                raise RuntimeError("Background acceptance timed out; inspect state, no automatic resubmission")
                            time.sleep(2)
                    write(out / (problem + ".json"), {"http_status": response.status_code,
                          "elapsed_s": time.perf_counter() - started, "response": payload})
                    if response.status_code != 200:
                        print("FAILED " + problem + " HTTP " + str(response.status_code), flush=True)
                        continue
                    aid = payload["attempt_id"]
                    write(out / (problem + "_traces.json"), client.get(f"/attempt/{aid}/traces").json())
                    write(out / (problem + "_record.json"), client.get(f"/attempt/{aid}").json())
                    if args.generate_review:
                        print("REVIEW_START " + problem, flush=True)
                        review_body = {"student_id": "media_accept_" + problem,
                                       "problem_id": problem, "attempt_id": aid,
                                       "run_id": payload["run_id"]}
                        review_started = time.perf_counter()
                        review = client.post("/video/review", json=review_body)
                        review_elapsed = time.perf_counter() - review_started
                        repeat_started = time.perf_counter()
                        repeat = client.post("/video/review", json=review_body)
                        write(out / (problem + "_review.json"), {
                            "http_status": review.status_code, "elapsed_s": review_elapsed,
                            "response": review.json(), "repeat_status": repeat.status_code,
                            "repeat_elapsed_s": time.perf_counter() - repeat_started,
                            "repeat_response": repeat.json()})
                        if review.status_code != 200 or repeat.status_code != 200:
                            raise RuntimeError("Separate review failed for " + problem)
                        timeline = (review.json().get("review") or {}).get("timeline")
                        if not isinstance(timeline, str) or not timeline.strip():
                            raise RuntimeError("Empty separate review must not pass acceptance for " + problem)
                        if (review.json().get("cached") is not False
                                or repeat.json().get("cached") is not True
                                or not review.json().get("review_input_fingerprint")
                                or review.json().get("review_input_fingerprint") != repeat.json().get("review_input_fingerprint")):
                            raise RuntimeError("Separate review cache integrity failed for " + problem)
                        print("REVIEW_DONE " + problem, flush=True)
                        if args.verify_review_revision:
                            run_dir = video.RUNS_DIR / payload["run_id"]
                            initial_material = (run_dir / "review_input.json").read_bytes()
                            initial_report = (run_dir / "review_report.json").read_bytes()
                            last = payload["rounds"][-1]
                            # Added trailing newline only: an explicit AI lifecycle edit,
                            # never an assertion that the OCR text matches handwriting.
                            # Strip logic would ignore that, so change harmless spacing instead.
                            corrected = last["answer"].replace("\n", "\n\n", 1)
                            if corrected == last["answer"]:
                                corrected = last["answer"].replace("=", " = ", 1)
                            if corrected == last["answer"]:
                                raise RuntimeError("No semantic-preserving format edit available")
                            correction = client.post("/video/correct_ocr", json={
                                "student_id": review_body["student_id"], "problem_id": problem,
                                "attempt_id": aid, "source_id": last["source_id"],
                                "base_revision": last["source_revision"], "corrected_text": corrected})
                            stale = client.post("/video/review", json=review_body)
                            fresh_start = time.perf_counter()
                            fresh = client.post("/video/review", json={**review_body, "refresh_input": True})
                            fresh_elapsed = time.perf_counter() - fresh_start
                            repeat_start = time.perf_counter()
                            fresh_repeat = client.post("/video/review", json={**review_body, "input_revision": 1})
                            write(out / (problem + "_review_revision.json"), {
                                "edit_provenance": "AI formatting-only lifecycle probe, not teacher/student ground truth",
                                "correction_status": correction.status_code, "correction": correction.json(),
                                "stale_status": stale.status_code, "stale": stale.json(),
                                "fresh_status": fresh.status_code, "fresh_elapsed_s": fresh_elapsed, "fresh": fresh.json(),
                                "repeat_status": fresh_repeat.status_code,
                                "repeat_elapsed_s": time.perf_counter() - repeat_start, "repeat": fresh_repeat.json()})
                            if (correction.status_code != 200 or stale.status_code != 409
                                    or fresh.status_code != 200 or fresh.json().get("input_revision") != 1
                                    or fresh.json().get("cached") is not False
                                    or fresh_repeat.status_code != 200 or fresh_repeat.json().get("cached") is not True
                                    or (run_dir / "review_input.json").read_bytes() != initial_material
                                    or (run_dir / "review_report.json").read_bytes() != initial_report):
                                raise RuntimeError("Review revision acceptance failed; inspect saved evidence")
                    checks.append({"problem": problem, "upload_removed": payload.get("upload_removed"),
                                   "snapshot_count": payload.get("snapshot_count"), "submitted": payload.get("submitted"),
                                   "coverage": payload.get("coverage"), "stop_reason": payload.get("teaching_stop_reason")})
                    print("DONE " + problem + " submitted=" + str(payload.get("submitted")), flush=True)
                # Decode sequentially: average-FPS seeking is unreliable for these VFR files.
                import cv2
                cap = cv2.VideoCapture(str(sources[0])) if not args.skip_photo else None
                ok, frame, photo_pts = False, None, None
                while cap is not None:
                    ok, frame = cap.read()
                    if not ok or cap.get(cv2.CAP_PROP_POS_MSEC) >= 74000:
                        break
                if cap is not None:
                    photo_pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    cap.release()
                if ok:
                    _, jpg = cv2.imencode(".jpg", frame)
                    (out / ("photo_" + manifest['videos'][0]['problem_id'] + "_near74.jpg")).write_bytes(jpg.tobytes())
                    photo = client.post("/video/analyze_image", data={"student_id": "media_accept_photo",
                        "problem_id": manifest['videos'][0]['problem_id'], "request_hint":"true", "include_review":"false"}, files={"file": ("photo.jpg", jpg.tobytes(), "image/jpeg")})
                    write(out / "photo.json", {"http_status": photo.status_code, "decoded_pts_sec": photo_pts, "response": photo.json()})
                    if args.photo_corrected_text and photo.status_code == 200 and photo.json().get("rounds"):
                        data = photo.json(); last = data["rounds"][-1]
                        req = {"student_id": "media_accept_photo", "problem_id": manifest['videos'][0]['problem_id'], "attempt_id": data["attempt_id"],
                               "source_id": last["source_id"], "base_revision": 0,
                               "corrected_text": args.photo_corrected_text}
                        correction = client.post("/video/correct_ocr", json=req)
                        repeat = client.post("/video/correct_ocr", json=req)
                        write(out / "photo_correction.json", {"transcription_provenance": "Explicit caller edit, not teacher gold",
                            "request": req, "http_status": correction.status_code, "response": correction.json(),
                            "repeat_status": repeat.status_code, "repeat_response": repeat.json()})
                        write(out / "photo_traces.json", client.get(f"/attempt/{data['attempt_id']}/traces").json())
                write(out / "checks.json", checks)
                if len(checks) != len(sources) or not all(c["upload_removed"] and c["submitted"] > 0 for c in checks):
                    raise RuntimeError("Video acceptance incomplete; inspect saved responses")
        finally:
            service.call_vision, service.run_pipeline = original_vision, original_pipeline
            write(out / "vision_usage.json", vision_calls)
            get_engine().dispose()
    print("COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
