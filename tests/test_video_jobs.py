"""Bounded jobs preserve progress and ownership without invoking external services."""
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from app.api import video, video_jobs
from app.services.video_jobs import VideoJobQueue
from app.services.video_progress import (progress_callback, report_progress, owned_upload_callback,
                                         register_owned_upload, perception_progress_callback)


def params(**kw):
    return {"student_id": "s", "problem_id": "p", "attempt_id": "", "every_sec": 20,
            "request_hint": False, "include_review": False, **kw}


def upload(content=b"video"):
    return UploadFile(filename="video.mp4", file=io.BytesIO(content))


def wait_idle(manager):
    # Bound waits so broken queue accounting fails a test instead of hanging CI.
    deadline = time.monotonic() + 3
    while manager.pending.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    assert manager.pending.unfinished_tasks == 0


def test_queue_returns_before_analysis_and_limits_total_admissions(tmp_path):
    started, release = threading.Event(), threading.Event()
    seen = []
    def runner(file, **kw):
        seen.append(file.file.read())
        report_progress("teaching", 1, 2, "snapshots", attempt_id="real-a")
        started.set()
        assert release.wait(3)
        return {"attempt_id": "real-a", "rounds": []}
    manager = VideoJobQueue(tmp_path, capacity=2, runner=runner)
    first = manager.submit(upload(), params(), 100)
    assert first["status"] == "queued"
    assert started.wait(2)
    running = manager.get(first["job_id"], "s", "p")
    assert running["status"] == "running" and running["phase"] == "teaching"
    assert running["attempt_id"] == "real-a"
    assert running["progress"] == {"completed": 1, "total": 2, "unit": "snapshots"}
    second = manager.submit(upload(b"second"), params(), 100)
    assert manager.get(second["job_id"], "s", "p")["status"] == "queued"
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(), params(), 100)
    assert error.value.status_code == 429
    release.set()
    wait_idle(manager)
    final = manager.get(first["job_id"], "s", "p")
    assert final["status"] == "succeeded" and final["result"]["attempt_id"] == "real-a"
    assert final["upload_removed"] is True
    assert seen == [b"video", b"second"]
    assert "params" not in final and "filename" not in final
    assert not list(tmp_path.glob("*/source.video"))


@pytest.mark.parametrize("content,maximum,status", [(b"", 100, 400), (b"long", 2, 413)])
def test_rejected_upload_releases_capacity_and_deletes_spool(tmp_path, content, maximum, status):
    manager = VideoJobQueue(tmp_path, capacity=1, runner=lambda *a, **kw: {"attempt_id": "a"})
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(content), params(), maximum)
    assert error.value.status_code == status
    assert not list(tmp_path.iterdir())
    good = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    assert manager.get(good["job_id"], "s", "p")["status"] == "succeeded"


@pytest.mark.parametrize("exception,status,message", [
    (HTTPException(409, "作答已结束"), 409, "作答已结束"),
    (ValueError("private implementation detail"), 502, "视频处理失败"),
])
def test_failure_keeps_partial_attempt_and_does_not_kill_worker(tmp_path, exception, status, message):
    runs = []
    def runner(file, **kw):
        runs.append(1)
        if len(runs) == 1:
            report_progress("teaching", 1, 3, "snapshots", attempt_id="partial")
            raise exception
        return {"attempt_id": "next"}
    manager = VideoJobQueue(tmp_path, runner=runner)
    first = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    failed = manager.get(first["job_id"], "s", "p")
    assert failed["status"] == "failed" and failed["result"] is None
    assert failed["error"]["status_code"] == status and message in failed["error"]["message"]
    assert "private" not in failed["error"]["message"]
    assert failed["attempt_id"] == "partial" and failed["upload_removed"] is True
    second = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    assert manager.get(second["job_id"], "s", "p")["status"] == "succeeded"


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_restart_marks_only_unfinished_jobs_interrupted_and_never_reruns(tmp_path, status):
    job_id = "a" * 32
    data = {"job_id": job_id, "student_id": "s", "problem_id": "p", "attempt_id": "partial",
            "status": status, "phase": "teaching", "progress": {"completed": 1, "total": 3, "unit": "snapshots"},
            "result": {"attempt_id": "partial"} if status == "succeeded" else None, "error": None}
    (tmp_path / f"{job_id}.json").write_text(json.dumps(data), encoding="utf-8")
    if status in {"queued", "running"}:
        folder = tmp_path / job_id
        folder.mkdir()
        (folder / "source.video").write_bytes(b"video")
    manager = VideoJobQueue(tmp_path, runner=lambda *a, **kw: pytest.fail("must never rerun on restart"))
    state = manager.get(job_id, "s", "p")
    if status in {"queued", "running"}:
        assert state["status"] == "interrupted" and state["spool_removed"] is True
        assert state["upload_removed"] is (status == "queued")
        assert state["analysis_upload_removed"] is (True if status == "queued" else None)
        assert "不会自动重跑" in state["error"]["message"]
        assert state["attempt_id"] == "partial"
    else:
        assert state == data
    assert manager.worker is None and manager.pending.empty()


def test_owner_id_and_corrupt_state_checks(tmp_path):
    manager = VideoJobQueue(tmp_path, runner=lambda *a, **kw: {"attempt_id": "a"})
    job = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    for student, problem in [("other", "p"), ("s", "other")]:
        with pytest.raises(HTTPException) as error:
            manager.get(job["job_id"], student, problem)
        assert error.value.status_code == 403
    for job_id, expected in [("../other", 422), ("b" * 32, 404)]:
        with pytest.raises(HTTPException) as error:
            manager.get(job_id, "s", "p")
        assert error.value.status_code == expected
    (tmp_path / f"{job['job_id']}.json").write_text("[]")
    with pytest.raises(HTTPException) as error:
        manager.get(job["job_id"], "s", "p")
    assert error.value.status_code == 409


def test_contextvar_hooks_are_isolated_between_threads_and_safe_on_storage_failure():
    received = []
    barrier = threading.Barrier(2)
    def request(label):
        token = progress_callback.set(lambda phase, progress: received.append((label, phase)))
        try:
            barrier.wait(timeout=2)
            report_progress(label)
        finally:
            progress_callback.reset(token)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(request, ["a", "b"]))
    report_progress("outside")
    assert sorted(received) == [("a", "a"), ("b", "b")]
    token = progress_callback.set(lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    try:
        report_progress("teaching", 1, 2)
    finally:
        progress_callback.reset(token)


def test_worker_survives_terminal_state_write_failure(tmp_path, monkeypatch):
    manager = VideoJobQueue(tmp_path, runner=lambda *a, **kw: {"attempt_id": "a"})
    original = manager._write
    failures = [1]
    def write(data):
        if data["status"] == "succeeded" and failures:
            failures.pop()
            raise OSError("disk full")
        original(data)
    monkeypatch.setattr(manager, "_write", write)
    manager.submit(upload(), params(), 100)
    wait_idle(manager)
    next_job = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    assert manager.get(next_job["job_id"], "s", "p")["status"] == "succeeded"
    assert manager.worker.is_alive()


def test_background_http_runs_real_video_endpoint_and_exposes_progress(tmp_path, monkeypatch):
    from app.vision import service
    entered, release = threading.Event(), threading.Event()
    manager = VideoJobQueue(tmp_path / "jobs")
    monkeypatch.setattr(video_jobs, "get_video_jobs", lambda: manager)
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "runs" / "uploads")
    targets = []
    monkeypatch.setattr(video, "_validate_media_target", lambda *args: targets.append(args))
    def pipeline(path, out_dir, *a, **kw):
        assert path.read_bytes() == b"real bytes"
        entered.set()
        assert release.wait(3)
        out_dir.mkdir(parents=True)
        (out_dir / "results.json").write_text(json.dumps({"frames": [
            {"timestamp_sec": 2, "text": "F=ma"}, {"timestamp_sec": 30, "text": "F=4N"}]}))
        return {"coverage": {}, "timings": {"decode_sec": 1}}
    monkeypatch.setattr(service, "run_pipeline", pipeline)
    monkeypatch.setattr(video, "analyze_attempt", lambda req: SimpleNamespace(model_dump=lambda: {
        "attempt_id": "real-attempt", "source_id": "source", "source_revision": 0, "finished": True}))
    monkeypatch.setattr(video, "_save_review_input", lambda *a: None)
    app = FastAPI()
    app.include_router(video_jobs.router)
    with TestClient(app) as client:
        response = client.post("/video/jobs", data={"student_id": "s", "problem_id": "p"},
                               files={"file": ("v.mp4", b"real bytes", "video/mp4")})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        assert entered.wait(2)
        url = f"/video/jobs/{job_id}"
        state = client.get(url, params={"student_id": "s", "problem_id": "p"}).json()
        assert state["status"] == "running" and state["phase"] == "perception"
        assert client.get(url, params={"student_id": "other", "problem_id": "p"}).status_code == 403
        release.set()
        wait_idle(manager)
        result = client.get(url, params={"student_id": "s", "problem_id": "p"}).json()
        assert result["status"] == "succeeded" and result["attempt_id"] == "real-attempt"
        assert result["result"]["submitted"] == 1
        assert result["result"]["remaining_snapshots"] == 1
        assert result["result"]["review"] is None
        assert result["upload_removed"] is True and result["result"]["upload_removed"] is True
        assert result["progress"] == {"completed": 1, "total": 2, "unit": "snapshots"}
    assert len(targets) == 2  # Admission and worker both revalidate ownership/lifecycle.


def test_restart_cleans_registered_inner_upload_and_spool(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    uploads = tmp_path / "runs" / "uploads"
    uploads.mkdir(parents=True)
    monkeypatch.setattr(video, "UPLOAD_DIR", uploads)
    job_id, name = "b" * 32, "a" * 32 + ".mp4"
    folder = root / job_id
    folder.mkdir()
    (folder / "source.video").write_bytes(b"original spool")
    (uploads / name).write_bytes(b"inner original")
    (root / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "student_id": "s", "problem_id": "p", "status": "running",
        "attempt_id": "partial", "analysis_upload_registered": True, "analysis_upload_name": name,
        "result": None, "phase": "teaching", "progress": {"completed": 1}}))
    manager = VideoJobQueue(root)
    state = manager.get(job_id, "s", "p")
    assert state["status"] == "interrupted"
    assert state["spool_removed"] is True and state["analysis_upload_removed"] is True
    assert state["upload_removed"] is True
    assert not (folder / "source.video").exists() and not (uploads / name).exists()
    assert "analysis_upload_name" not in state


def test_registration_failure_prevents_inner_file_creation(tmp_path, monkeypatch):
    uploads = tmp_path / "runs" / "uploads"
    monkeypatch.setattr(video, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(video, "_validate_media_target", lambda *a: None)
    manager = VideoJobQueue(tmp_path / "jobs")
    original = manager._write
    def write(data):
        if data.get("analysis_upload_registered"):
            raise OSError("registration storage failed")
        original(data)
    monkeypatch.setattr(manager, "_write", write)
    job = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    state = manager.get(job["job_id"], "s", "p")
    assert state["status"] == "failed" and state["upload_removed"] is True
    assert not list(uploads.glob("*"))


@pytest.mark.parametrize("filename", ["../../outside.mp4", "C:\\outside.mp4", "not-random.mp4"])
def test_recovery_rejects_inner_path_escape_without_deleting_outside(tmp_path, monkeypatch, filename):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(video, "UPLOAD_DIR", uploads)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"keep")
    root = tmp_path / "jobs"
    root.mkdir()
    job_id = "a" * 32
    (root / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id, "student_id": "s", "problem_id": "p", "status": "running",
        "analysis_upload_registered": True, "analysis_upload_name": filename, "result": None}))
    manager = VideoJobQueue(root)
    state = manager.get(job_id, "s", "p")
    assert state["analysis_upload_removed"] is False and state["upload_removed"] is False
    assert outside.read_bytes() == b"keep"


def test_spool_removal_does_not_cover_explicit_inner_cleanup_failure(tmp_path):
    manager = VideoJobQueue(tmp_path, runner=lambda *a, **kw: {"attempt_id": "a", "upload_removed": False})
    job = manager.submit(upload(), params(), 100)
    wait_idle(manager)
    state = manager.get(job["job_id"], "s", "p")
    assert state["spool_removed"] is True
    assert state["analysis_upload_removed"] is False and state["upload_removed"] is False
    assert state["result"]["upload_removed"] is False


def test_perception_callback_captures_request_context_for_pool_and_counts_do_not_regress(tmp_path):
    manager = VideoJobQueue(tmp_path)
    job_id = "a" * 32
    manager._write({"job_id": job_id, "student_id": "s", "problem_id": "p", "status": "running",
                    "phase": "upload", "progress": {"completed": 0, "total": None, "unit": "stage"}})
    token = progress_callback.set(lambda phase, progress: manager._progress(job_id, phase, progress))
    try:
        callback = perception_progress_callback()
    finally:
        progress_callback.reset(token)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(callback, "extracting", "decode").result()
        assert manager.get(job_id, "s", "p")["phase"] == "extracting"
        pool.submit(callback, "analyzing", "ocr", analyze_done=3, analyze_total=5).result()
        pool.submit(callback, "analyzing", "late", analyze_done=2, analyze_total=5).result()
    state = manager.get(job_id, "s", "p")
    assert state["phase"] == "recognizing" and state["progress"] == {"completed": 3, "total": 5, "unit": "frames"}
    callback("exporting", "save")
    assert manager.get(job_id, "s", "p")["phase"] == "exporting"


def test_http_rejects_target_before_spooling(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "_validate_media_target", lambda *a: (_ for _ in ()).throw(HTTPException(409, "closed")))
    monkeypatch.setattr(video_jobs, "get_video_jobs", lambda: pytest.fail("must not create a queue"))
    app = FastAPI()
    app.include_router(video_jobs.router)
    with TestClient(app) as client:
        result = client.post("/video/jobs", data={"student_id": "s", "problem_id": "p"},
                             files={"file": ("v.mp4", b"video")})
    assert result.status_code == 409


# ================================================================ 任务找回（契约 §2.2）
#
# 一个上传编号 = **一次上传动作**。同一编号重复提交必须是**幂等**的：
# 返回原任务，**不产生第二次教学或模型调用**。
# 编号↔任务的映射**持久化**（进程重启后仍可查），且**只有任务文件一个真相源**。

RID = "11111111-2222-3333-4444-555555555555"


def idle_manager(tmp_path, **kw):
    """一个不会真跑模型的队列；`seen` 记下实际跑了几次。"""
    seen = []
    def runner(*a, **kw2):
        seen.append(1)
        return {"attempt_id": "a"}
    manager = VideoJobQueue(tmp_path, runner=runner, **kw)
    return manager, seen


def test_same_request_id_and_content_returns_the_original_job(tmp_path):
    manager, seen = idle_manager(tmp_path)
    first = manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    again = manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    assert again["job_id"] == first["job_id"], "收到的是新任务，不是原任务"
    assert len(seen) == 1, "重复提交产生了第二次模型调用"
    assert len(list(tmp_path.glob("*.json"))) == 1, "建了第二个任务"


def test_same_request_id_with_different_content_is_a_conflict(tmp_path):
    manager, _ = idle_manager(tmp_path)
    manager.submit(upload(b"video"), params(), 100, RID)
    wait_idle(manager)
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(b"OTHER"), params(), 100, RID)
    assert error.value.status_code == 409
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_same_request_id_with_different_params_is_a_conflict(tmp_path):
    manager, _ = idle_manager(tmp_path)
    manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(), params(problem_id="other"), 100, RID)
    assert error.value.status_code == 409


def test_different_params_are_rejected_without_even_reading_the_upload(tmp_path):
    """**顺序是刻意的**：参数不同立刻 409，连上传都不读（省掉一次整传）。"""
    manager, _ = idle_manager(tmp_path)
    manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    stream = io.BytesIO(b"x" * 5000)
    probe = UploadFile(filename="v.mp4", file=stream)
    with pytest.raises(HTTPException):
        manager.submit(probe, params(problem_id="other"), 100, RID)
    assert stream.tell() == 0, "参数不同却把上传读了一遍"


def test_a_duplicate_submit_succeeds_even_when_the_queue_is_full(tmp_path):
    """⚠️ 队列满时重复提交**必须拿回原任务**，而不是一个会误导人的 429。

    队列满说明"还有别的任务在跑"，与"这次上传是不是重复"无关。
    """
    release = threading.Event()
    def runner(*a, **kw):
        assert release.wait(3)
        return {"attempt_id": "a"}
    manager = VideoJobQueue(tmp_path, capacity=1, runner=runner)
    first = manager.submit(upload(), params(), 100, RID)      # 占住唯一的槽
    assert manager.get(first["job_id"], "s", "p")["status"] in {"queued", "running"}
    again = manager.submit(upload(), params(), 100, RID)      # 重复 —— 不该 429
    assert again["job_id"] == first["job_id"]
    release.set()
    wait_idle(manager)


def test_concurrent_duplicates_create_exactly_one_job(tmp_path):
    """两个并发请求带同一编号 ⇒ 晚到的那个**回滚自己**，回放先到的。"""
    manager, seen = idle_manager(tmp_path, capacity=4)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in
                   [pool.submit(manager.submit, upload(), params(), 100, RID) for _ in range(2)]]
    wait_idle(manager)
    assert len({r["job_id"] for r in results}) == 1, "并发重复提交建出了两个任务"
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(seen) == 1


def test_concurrent_duplicates_with_different_params_are_a_conflict(tmp_path):
    """⚠️ **加锁后的那次检查也必须比参数。**

    只比文件哈希的话，"同编号 + 同文件 + **不同参数**"的并发请求会**拿回原任务**，
    而它本该是 409（它要的是另一次处理）。预查那条路本来就比参数 ——
    所以这个漏**只在并发这一条路上**，最难发现。

    做法：把 `_find_by_request` 的**第一次**调用挡掉（模拟"两个请求都没预查到"），
    逼它走到加锁后那条检查。
    """
    manager, _ = idle_manager(tmp_path, capacity=4)
    original = manager._find_by_request
    calls = {"n": 0}

    def hide_first(request_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return None              # 预查扑空 ⇒ 走加锁后那条路
        return original(request_id)

    manager._find_by_request = hide_first
    first = manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(), params(problem_id="other"), 100, RID)
    assert error.value.status_code == 409, "同编号不同参数的并发提交没有报 409"
    assert len(list(tmp_path.glob("*.json"))) == 1, "冲突的那次也建了任务"


def test_malformed_request_id_is_rejected(tmp_path):
    manager, _ = idle_manager(tmp_path)
    with pytest.raises(HTTPException) as error:
        manager.submit(upload(), params(), 100, "not-a-uuid")
    assert error.value.status_code == 422
    assert not list(tmp_path.glob("*.json"))


def test_the_mapping_survives_a_restart(tmp_path):
    """契约 §2.2：编号↔任务映射**必须持久化**，进程重启后仍可查。"""
    manager, _ = idle_manager(tmp_path)
    created = manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    restarted = VideoJobQueue(tmp_path, runner=lambda *a, **kw: {"attempt_id": "a"})
    found = restarted.get_by_request(RID, "s", "p")
    assert found["job_id"] == created["job_id"]
    # 重启后**不自动重跑**（原有语义不变）
    assert found["status"] == "succeeded"


def test_by_request_hides_existence_from_other_owners(tmp_path):
    """⚠️ 归属不符 ⇒ **404，不是 403**。403 等于说漏"这个编号存在"。"""
    manager, _ = idle_manager(tmp_path)
    created = manager.submit(upload(), params(), 100, RID)
    wait_idle(manager)
    assert manager.get_by_request(RID, "s", "p")["job_id"] == created["job_id"]
    for student, problem in (("other", "p"), ("s", "other")):
        with pytest.raises(HTTPException) as error:
            manager.get_by_request(RID, student, problem)
        assert error.value.status_code == 404, "不该用 403 泄露存在性"
    with pytest.raises(HTTPException) as error:
        manager.get_by_request("99999999-2222-3333-4444-555555555555", "s", "p")
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        manager.get_by_request("not-a-uuid", "s", "p")
    assert error.value.status_code == 422


def test_a_submit_without_a_request_id_behaves_as_before(tmp_path):
    """不带编号 ⇒ 老行为不变（两次提交就是两个任务）。"""
    manager, _ = idle_manager(tmp_path)
    first = manager.submit(upload(), params(), 100)
    second = manager.submit(upload(), params(), 100)
    assert first["job_id"] != second["job_id"]
    assert first.get("client_request_id") is None


def test_http_round_trip_for_idempotent_submit_and_lookup(tmp_path, monkeypatch):
    """走真实 HTTP：提交带编号、重复提交拿回原任务、按编号查得到。"""
    monkeypatch.setattr(video, "_validate_media_target", lambda *a, **kw: None)
    monkeypatch.setattr(video, "MAX_UPLOAD_BYTES", 10 ** 6)
    manager, seen = idle_manager(tmp_path / "jobs")
    monkeypatch.setattr(video_jobs, "get_video_jobs", lambda: manager)
    app = FastAPI()
    app.include_router(video_jobs.router)
    form = {"student_id": "s", "problem_id": "p", "client_request_id": RID}
    with TestClient(app) as client:
        first = client.post("/video/jobs", data=form, files={"file": ("v.mp4", b"video")})
        assert first.status_code == 200, first.text
        again = client.post("/video/jobs", data=form, files={"file": ("v.mp4", b"video")})
        assert again.status_code == 200 and again.json()["job_id"] == first.json()["job_id"]
        conflict = client.post("/video/jobs", data=form, files={"file": ("v.mp4", b"OTHER")})
        assert conflict.status_code == 409
        found = client.get(f"/video/jobs/by-request/{RID}",
                           params={"student_id": "s", "problem_id": "p"})
        assert found.status_code == 200 and found.json()["job_id"] == first.json()["job_id"]
        assert client.get(f"/video/jobs/by-request/{RID}",
                          params={"student_id": "x", "problem_id": "p"}).status_code == 404
        bad = client.post("/video/jobs", data={**form, "client_request_id": "nope"},
                          files={"file": ("v.mp4", b"video")})
        assert bad.status_code == 422
    wait_idle(manager)
    assert len(seen) == 1, "HTTP 层重复提交产生了第二次模型调用"
