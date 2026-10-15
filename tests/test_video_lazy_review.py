"""Deferred video review uses immutable server inputs and never advances teaching."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient

from app.api import video
from app.services import attempt_store, problem_repository, review_builder
from app.vision import service
from app.services import video_review_versions


def _report_payload(timeline="report"):
    return {"timeline": timeline, "sticking_points": [], "key_concept": "momentum",
            "next_step": "check direction", "encouragement": "continue", "answer_check": None,
            "render_policy": "legacy", "canonicalized_fields": []}


def _report_path(root):
    reports = list((root / "review_reports").glob("*.json"))
    return reports[0] if reports else root / "review_report.json"


@pytest.fixture
def review_env(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(video, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(video, "UPLOAD_DIR", tmp_path / "runs" / "uploads")
    state = SimpleNamespace(media_sources=[{"source_id": "src", "revision": 0}],
                            timeline=[{}], graph_version="graph-1", graph_snapshot={"nodes": []})
    record = SimpleNamespace(student_id="s", problem_id="p", state=state, status="finished")
    monkeypatch.setattr(attempt_store, "get_attempt_store", lambda: SimpleNamespace(
        load_attempt=lambda key: record if key == "a" else None))
    monkeypatch.setattr(problem_repository, "get_problem_repository", lambda: SimpleNamespace(
        get_problem=lambda _: SimpleNamespace(statement="original problem", knowledge_points=["momentum"])))
    run_id = "a" * 32
    root = video.RUNS_DIR / run_id
    video._save_review_input(root, "s", "p", "a", [
        {"answer": "original OCR", "source_id": "src", "source_revision": 0, "graph_version": "graph-1"}])
    monkeypatch.setattr(video_review_versions, "generation_context", lambda: ({"model": "test"}, "test-generation", None))
    calls = []
    def generate(problem, rounds, **_kwargs):
        calls.append((problem.statement, rounds[0]["answer"]))
        return SimpleNamespace(model_dump=lambda: _report_payload())
    monkeypatch.setattr(review_builder, "review_trajectory", generate)
    req = video.VideoReviewRequest(student_id="s", problem_id="p", attempt_id="a", run_id=run_id)
    return req, state, root, calls


def test_finished_attempt_review_cached_across_memory_reset(review_env):
    req, _, _, calls = review_env
    first = video.review_video(req)
    from app.services import graph_versions
    graph_versions._LOCKS.clear()  # Simulate process-local lock loss; persisted report remains authoritative.
    second = video.review_video(req)
    assert first["cached"] is False and second["cached"] is True
    assert first["review_input_fingerprint"] == second["review_input_fingerprint"]
    assert calls == [("original problem", "original OCR")]
    assert first["review_scope"] == "video_run_snapshot"
    assert first["graph_version"] == "graph-1"


def test_concurrent_review_is_idempotent(review_env):
    req, _, _, calls = review_env
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: video.review_video(req), range(4)))
    assert len(calls) == 1
    assert sum(not r["cached"] for r in responses) == 1


@pytest.mark.parametrize("change", ["revision", "timeline"])
def test_corrected_or_extended_input_rejected_even_with_cache(review_env, change):
    req, state, _, calls = review_env
    video.review_video(req)
    if change == "revision":
        state.media_sources[0]["revision"] = 1
    else:
        state.timeline.append({})
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    assert len(calls) == 1


def test_failure_can_retry_without_mutating_attempt(review_env, monkeypatch):
    req, state, root, calls = review_env
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: None)
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 502
    assert not (root / "review_report.json").exists()
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(model_dump=lambda: _report_payload("retry")))
    assert video.review_video(req)["review"]["timeline"] == "retry"
    assert state.timeline == [{}]


def test_endpoint_rejects_client_rounds_and_bad_paths(review_env):
    req, _, _, calls = review_env
    app = FastAPI()
    app.include_router(video.router)
    with TestClient(app) as client:
        data = req.model_dump()
        assert client.post("/video/review", json={**data, "rounds": [{"answer": "forged"}]}).status_code == 422
        assert client.post("/video/review", json={**data, "run_id": "../uploads"}).status_code == 422
        assert client.post("/video/review", json={**data, "student_id": "other"}).status_code == 403
        assert client.post("/video/review", json={**data, "attempt_id": "missing"}).status_code == 404
        assert client.post("/video/review", json={**data, "run_id": "b" * 32}).status_code == 404
        assert client.post("/video/review", json=data).status_code == 200
    assert len(calls) == 1


def test_run_ownership_and_corruption_rejected(review_env):
    req, _, root, _ = review_env
    path = root / "review_input.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["input"]["student_id"] = "other"
    source["fingerprint"] = video._fingerprint(source["input"])
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 403
    path.write_text("{", encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409


def test_video_default_does_not_call_review_and_explicit_flag_does(review_env, monkeypatch):
    req, _, _, calls = review_env
    monkeypatch.setattr(video, "_validate_media_target", lambda *_, **_kwargs: None)
    def pipeline(_path, out, *_args, **_kwargs):
        out.mkdir(parents=True)
        (out / "results.json").write_text("{}")
        return {"coverage": {}, "timings": {"decode_sec": 1}}
    monkeypatch.setattr(service, "run_pipeline", pipeline)
    monkeypatch.setattr(video, "_snapshots_from", lambda *_, **_kwargs: [{"ts": 2, "text": "original OCR"}])
    monkeypatch.setattr(video, "analyze_attempt", lambda *_, **_kwargs: SimpleNamespace(model_dump=lambda: {
        "attempt_id": "a", "finished": True, "source_id": "src", "source_revision": 0}))
    def analyze(**kwargs):
        return video.analyze_video(UploadFile(filename="video.mp4", file=io.BytesIO(b"video")),
            problem_id="p", student_id="s", attempt_id="a", every_sec=20, request_hint=False, **kwargs)
    first = analyze()  # Direct Python default is a Form object: must not trigger review.
    # 契约 §2.1：没请求 ⇒ `not_requested`（**不是 `failed`，也不是已废弃的 `available`**）；
    # 材料还在、可以稍后生成 ⇒ `review_available`（与生成状态**正交**）。
    assert first["review"] is None and first["review_status"] == "not_requested"
    assert first["review_available"] is True
    assert first["perception_timings"] == {"decode_sec": 1}
    assert calls == []
    assert analyze(include_review=True)["review_status"] == "ready"
    assert len(calls) == 1


def test_correction_during_generation_cannot_publish_old_review(review_env, monkeypatch):
    req, state, root, _ = review_env
    def generate(*_, **_kwargs):
        state.media_sources[0]["revision"] = 1
        return SimpleNamespace(model_dump=lambda: _report_payload("stale"))
    monkeypatch.setattr(review_builder, "review_trajectory", generate)
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    assert not (root / "review_report.json").exists()


def test_report_fingerprint_mismatch_recomputes_from_original_problem(review_env, monkeypatch):
    req, _, root, calls = review_env
    video.review_video(req)
    cache = _report_path(root)
    report = json.loads(cache.read_text(encoding="utf-8"))
    report["review_input_fingerprint"] = "wrong"
    cache.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(problem_repository, "get_problem_repository", lambda: SimpleNamespace(
        get_problem=lambda _: SimpleNamespace(statement="changed bank", knowledge_points=[])))
    assert video.review_video(req)["cached"] is False
    assert calls == [("original problem", "original OCR")] * 2


@pytest.mark.parametrize("broken", [[], None, "json string", {}, {"input": []}, {"input": None}])
def test_valid_json_with_wrong_input_envelope_is_409(review_env, broken):
    req, _, root, calls = review_env
    (root / "review_input.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("student_id", []), ("problem_id", None), ("attempt_id", ""),
    ("problem", []), ("problem", {}),
    ("problem", {"statement": 1, "knowledge_points": []}),
    ("problem", {"statement": "p", "knowledge_points": "momentum"}),
    ("problem", {"statement": "p", "knowledge_points": [None]}),
    ("rounds", None), ("rounds", {}), ("rounds", [None]), ("rounds", [{}]),
    ("source_revisions", []), ("source_revisions", {"src": "0"}),
    ("source_revisions", {"src": True}), ("source_revisions", {"src": -1}),
    ("graph_version", None), ("graph_snapshot", []),
    ("timeline_length", "1"), ("timeline_length", True), ("timeline_length", -1),
    ("scope", None),
])
def test_review_input_field_types_are_409_even_with_matching_fingerprint(review_env, field, value):
    req, _, root, calls = review_env
    path = root / "review_input.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["input"][field] = value
    saved["fingerprint"] = video._fingerprint(saved["input"])
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    assert calls == []


@pytest.mark.parametrize("field", ["student_id", "problem_id", "attempt_id", "problem", "rounds",
                                   "source_revisions", "graph_version", "graph_snapshot", "timeline_length", "scope"])
def test_missing_review_input_fields_are_409(review_env, field):
    req, _, root, calls = review_env
    path = root / "review_input.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["input"][field]
    saved["fingerprint"] = video._fingerprint(saved["input"])
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    assert calls == []


@pytest.mark.parametrize("broken", [[], None, "json string", {},
    {"review": []}, {"review": "report"}, {"review_elapsed_sec": None}])
def test_wrong_cache_shape_is_a_cache_miss(review_env, broken):
    req, _, root, calls = review_env
    cache = _report_path(root)
    if isinstance(broken, dict):
        broken = {"review_input_fingerprint": json.loads((root / "review_input.json").read_text(encoding="utf-8"))["fingerprint"], **broken}
    cache.write_text(json.dumps(broken), encoding="utf-8")
    result = video.review_video(req)
    assert result["cached"] is False
    assert result["review"] == _report_payload()
    assert len(calls) == 1
    assert video.review_video(req)["cached"] is True


@pytest.mark.parametrize("elapsed", [True, "1", -1, float("inf"), float("nan")])
def test_invalid_cached_elapsed_does_not_escape_as_invalid_response(review_env, elapsed):
    req, _, root, calls = review_env
    video.review_video(req)
    path = _report_path(root)
    cache = json.loads(path.read_text(encoding="utf-8"))
    cache["review_elapsed_sec"] = elapsed
    path.write_text(json.dumps(cache), encoding="utf-8")
    assert video.review_video(req)["cached"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("timeline", ["", "  ", None, []])
def test_old_empty_report_cache_is_recomputed(review_env, timeline):
    req, _, root, calls = review_env
    video.review_video(req)
    path = _report_path(root)
    cache = json.loads(path.read_text(encoding="utf-8"))
    cache["review"]["timeline"] = timeline
    path.write_text(json.dumps(cache), encoding="utf-8")
    result = video.review_video(req)
    assert result["cached"] is False
    assert result["review"]["timeline"] == "report"
    assert len(calls) == 2


@pytest.mark.parametrize("timeline", ["", "  ", None, []])
def test_empty_generated_report_is_502_not_cached_and_can_retry(review_env, monkeypatch, timeline):
    req, _, root, _ = review_env
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(
        model_dump=lambda: _report_payload(timeline)))
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 502
    assert not (root / "review_report.json").exists()
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(
        model_dump=lambda: _report_payload("real report")))
    assert video.review_video(req)["review"]["timeline"] == "real report"
    assert video.review_video(req)["cached"] is True


@pytest.mark.parametrize("field,value", [
    ("sticking_points", [None]), ("sticking_points", "bad"), ("sticking_points", [{}]),
    ("sticking_points", [{"where": "step", "what": None, "why": "reason"}]),
    ("answer_check", "bad"), ("answer_check", {}),
    ("answer_check", {"student_answer": "1", "derived_from_work": "1", "consistent": []}),
    ("key_concept", []), ("next_step", None), ("encouragement", 42),
])
@pytest.mark.parametrize("where", ["cache", "generated"])
def test_nonempty_timeline_with_invalid_report_fields_is_never_ready(review_env, monkeypatch, field, value, where):
    req, _, root, calls = review_env
    payload = {**_report_payload("nonempty summary"), field: value}
    if where == "cache":
        video.review_video(req)
        path = _report_path(root)
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["review"] = payload
        path.write_text(json.dumps(cache), encoding="utf-8")
        result = video.review_video(req)
        assert result["cached"] is False
        assert result["review"] == _report_payload()
        assert len(calls) == 2
    else:
        monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(model_dump=lambda: payload))
        with pytest.raises(HTTPException) as error:
            video.review_video(req)
        assert error.value.status_code == 502
        assert not (root / "review_report.json").exists()


@pytest.mark.parametrize("field", ["sticking_points", "key_concept", "next_step", "encouragement"])
@pytest.mark.parametrize("where", ["cache", "generated"])
def test_missing_required_report_fields_are_rejected(review_env, monkeypatch, field, where):
    req, _, root, calls = review_env
    payload = _report_payload("nonempty summary")
    del payload[field]
    if where == "cache":
        video.review_video(req)
        path = _report_path(root)
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["review"] = payload
        path.write_text(json.dumps(cache), encoding="utf-8")
        result = video.review_video(req)
        assert result["cached"] is False and result["review"] == _report_payload()
    else:
        monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(model_dump=lambda: payload))
        with pytest.raises(HTTPException) as error:
            video.review_video(req)
        assert error.value.status_code == 502
        assert not (root / "review_report.json").exists()


# ================================================================ 引用关口：生成与缓存两条路
#
# ⚠️ **本轮堵的洞**：旧版只有**生成**路径查 `dangling_reference`，
# 缓存路径只查形状 ⇒ 一份**旧代码写下的**（或从别处来的）带坏引用的报告，
# 会在缓存里被一路放行、当成功结果返回。
# ⇒ 现在两条路共用 `_blocking_reference_defects`（**重算**，不读缓存里的 validation）。

def _v3_payload(evidence_ids, review_schema_version="v3+prompt-v3"):
    payload = _report_payload()
    payload.update({
        "review_schema_version": review_schema_version,
        "evidence": [{"id": "E1.1", "source": "student_answer", "source_id": None,
                      "revision": None, "round": 1, "ts": 1.0, "text": "I = 1.98e4"}],
        "findings": [{"category": "observed_error", "where": "方向", "what": "写反了",
                      "why": "", "downstream": "", "evidence_ids": list(evidence_ids)}],
        # 故意写一个「校验通过」—— 判据必须**重算**，不能被它骗过去
        "validation": {"valid": True, "scope": "结构与引用校验", "validator_version": "v3",
                       "issues": [], "evidence_count": 1},
    })
    return payload


def test_generated_report_with_a_dangling_reference_is_not_cached(review_env, monkeypatch):
    """生成路径：引用了不存在的证据 ⇒ 502，**不写缓存**（契约⑦）。"""
    req, _, root, _ = review_env
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *_, **_kwargs: SimpleNamespace(
        model_dump=lambda: _v3_payload(["E9.9"])))
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 502
    assert not (root / "review_report.json").exists()
    assert not list((root / "review_reports").glob("*.json"))


def test_cached_report_with_a_dangling_reference_is_not_served(review_env):
    """**缓存路径要有同等检查** —— 旧代码写下的坏报告不能一路放行。"""
    req, _, root, calls = review_env
    video.review_video(req)                       # 先生成一份好的并落缓存
    assert len(calls) == 1
    cache = _report_path(root)
    saved = json.loads(cache.read_text(encoding="utf-8"))
    saved["review"] = _v3_payload(["E9.9"])       # 把缓存改成「引用指不回去」
    cache.write_text(json.dumps(saved), encoding="utf-8")

    result = video.review_video(req)
    assert result["cached"] is False, "坏引用被当成缓存命中放行了"
    assert len(calls) == 2, "应当重新生成"


def test_a_cached_report_claiming_an_error_without_evidence_is_not_served(review_env):
    """**错误指控没有依据**也是缺陷（`unsupported_claim`），同样不能放行。

    这正是本轮要堵的那个洞：`valid=true` 曾经**不能**保证每条指控都有依据。
    """
    req, _, root, calls = review_env
    video.review_video(req)
    cache = _report_path(root)
    saved = json.loads(cache.read_text(encoding="utf-8"))
    saved["review"] = _v3_payload([])             # 错误指控，一个引用都没有
    cache.write_text(json.dumps(saved), encoding="utf-8")

    assert video.review_video(req)["cached"] is False
    assert len(calls) == 2


def test_a_cached_report_whose_validation_lies_is_still_caught(review_env):
    """⚠️ 判据**重算**：缓存里写着 `valid: true` 也救不了它。"""
    req, _, root, calls = review_env
    video.review_video(req)
    cache = _report_path(root)
    saved = json.loads(cache.read_text(encoding="utf-8"))
    lying = _v3_payload(["E9.9"])
    assert lying["validation"]["valid"] is True   # 它在说谎
    saved["review"] = lying
    cache.write_text(json.dumps(saved), encoding="utf-8")

    assert video.review_video(req)["cached"] is False
    assert len(calls) == 2


def test_v1_v2_cached_reports_are_still_served(review_env):
    """没有 evidence/findings 的旧报告**照常可读** —— 不伪装成带证据报告，
    但也不该因为「查不了引用」就被判坏。"""
    req, _, root, calls = review_env
    assert video.review_video(req)["cached"] is False
    assert video.review_video(req)["cached"] is True
    assert len(calls) == 1
