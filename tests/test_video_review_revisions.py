"""Corrected OCR is a new server-owned material; generation provenance affects caches."""
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import video
from app.services import review_builder, video_review_versions as versions
from test_video_lazy_review import review_env, _report_payload


def correct(state, text="corrected OCR"):
    source = state.media_sources[0]
    base = source["revision"]
    source["revision"] += 1
    revision = source["revision"]
    analysis = {"source_id": "src", "source_revision": revision, "current_node": "S2",
                "hint": "updated explanation", "stuck_score": 0.0}
    source.setdefault("revisions", []).append({"revision": revision, "text": text, "analysis": analysis})
    state.timeline.append({"event_type": "ocr_correction",
                           "content": f"source=src; revision={revision}; supersedes={base}"})


def test_corrected_material_is_explicit_version_and_preserves_original(review_env):
    req, state, root, calls = review_env
    first = video.review_video(req)
    original_input = (root / "review_input.json").read_bytes()
    original_report = (root / "review_report.json").read_bytes()
    correct(state)
    with pytest.raises(HTTPException) as error:
        video.review_video(req)
    assert error.value.status_code == 409
    refreshed = video.review_video(req.model_copy(update={"refresh_input": True}))
    assert refreshed["input_revision"] == 1
    assert refreshed["review_source_revisions"] == {"src": 1}
    assert refreshed["review_input_fingerprint"] != first["review_input_fingerprint"]
    assert calls == [("original problem", "original OCR"), ("original problem", "corrected OCR")]
    assert (root / "review_input.json").read_bytes() == original_input
    assert (root / "review_report.json").read_bytes() == original_report
    saved = json.loads((root / "review_inputs" / "1.json").read_text(encoding="utf-8"))
    assert saved["input"]["rounds"][0]["current_node"] == "S2"
    assert saved["input"]["rounds"][0]["hint"] == "updated explanation"
    current_req = req.model_copy(update={"input_revision": 1})
    assert video.review_video(current_req)["cached"] is True
    assert video.review_video(current_req.model_copy(update={"refresh_input": True}))["input_revision"] == 1
    with pytest.raises(HTTPException):
        video.review_video(req.model_copy(update={"refresh_input": True}))
    assert len(list((root / "review_reports").glob("*.json"))) == 2


@pytest.mark.parametrize("independent", ["submit", "hint_request", "source", "rewrite_prefix", "unlogged_revision"])
def test_refresh_rejects_unverifiable_or_independent_input(review_env, independent):
    req, state, root, calls = review_env
    correct(state)
    if independent == "source":
        state.media_sources.append({"source_id": "other", "revision": 0})
    elif independent == "rewrite_prefix":
        state.timeline[0] = {"content": "changed"}
    elif independent == "unlogged_revision":
        state.media_sources[0]["revision"] = 2
    else:
        state.timeline.append({"event_type": independent, "content": "new input"})
    with pytest.raises(HTTPException) as error:
        video.review_video(req.model_copy(update={"refresh_input": True}))
    assert error.value.status_code == 409
    assert calls == []
    assert not (root / "review_inputs").exists()


def test_failed_refresh_keeps_client_revision_retryable(review_env, monkeypatch):
    req, state, root, _ = review_env
    correct(state)
    refreshed_req = req.model_copy(update={"refresh_input": True})
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *a, **kw: None)
    with pytest.raises(HTTPException) as error:
        video.review_video(refreshed_req)
    assert error.value.status_code == 502
    assert not (root / "review_inputs" / "1.json").exists()
    monkeypatch.setattr(review_builder, "review_trajectory", lambda *a, **kw: SimpleNamespace(
        model_dump=lambda: _report_payload()))
    assert video.review_video(refreshed_req)["input_revision"] == 1


def test_second_correction_during_refreshed_generation_publishes_nothing(review_env, monkeypatch):
    req, state, root, _ = review_env
    correct(state)
    def generate(*a, **kw):
        correct(state, "changed again")
        return SimpleNamespace(model_dump=lambda: _report_payload())
    monkeypatch.setattr(review_builder, "review_trajectory", generate)
    with pytest.raises(HTTPException) as error:
        video.review_video(req.model_copy(update={"refresh_input": True}))
    assert error.value.status_code == 409
    assert not (root / "review_inputs" / "1.json").exists()
    assert not (root / "review_reports").exists()


def test_multiple_corrections_keep_material_versions_distinct(review_env):
    req, state, root, calls = review_env
    correct(state)
    video.review_video(req.model_copy(update={"refresh_input": True}))
    correct(state, "second correction")
    current_req = req.model_copy(update={"input_revision": 1, "refresh_input": True})
    assert video.review_video(current_req)["input_revision"] == 2
    assert sorted(p.stem for p in (root / "review_inputs").glob("*.json")) == ["1", "2"]
    assert calls[-1] == ("original problem", "second correction")


@pytest.mark.parametrize("missing", ["timeline_snapshot", "media_source_ids"])
def test_legacy_material_still_reads_but_cannot_be_refreshed(review_env, missing):
    req, state, root, _ = review_env
    path = root / "review_input.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["input"][missing]
    saved["fingerprint"] = video._fingerprint(saved["input"])
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert video.review_video(req)["input_revision"] == 0
    correct(state)
    with pytest.raises(HTTPException) as error:
        video.review_video(req.model_copy(update={"refresh_input": True}))
    assert error.value.status_code == 409
    assert "完整历史" in error.value.detail


def test_generation_change_has_new_cache_and_old_generation_is_retained(review_env, monkeypatch):
    req, _, root, calls = review_env
    initial = video.review_video(req)
    original_report = (root / "review_report.json").read_bytes()
    monkeypatch.setattr(versions, "generation_context", lambda: ({"model": "changed"}, "new-generation", None))
    result = video.review_video(req)
    assert result["cached"] is False
    assert result["review_input_fingerprint"] == initial["review_input_fingerprint"]
    assert result["generation_fingerprint"] != initial["generation_fingerprint"]
    assert video.review_video(req)["cached"] is True
    assert len(calls) == 2
    assert (root / "review_report.json").read_bytes() == original_report
    assert len(list((root / "review_reports").glob("*.json"))) == 2


@pytest.mark.parametrize("changed", ["model", "base_url", "temperature", "max_tokens", "thinking", "prompt", "implementation"])
def test_actual_generation_identity_and_pinned_prompt(monkeypatch, changed):
    from app.services import llm
    from app.agent import prompt_registry
    calls = []
    client = SimpleNamespace(name="provider", kind="openai_compatible", model="one",
                             base_url="https://user:secret@gateway.test/v1?api_key=secret",
                             temperature=0.2, timeout=30, max_retries=2, max_tokens=None,
                             chat_structured_resilient=lambda *args: calls.append(args))
    prompt = ["first prompt"]
    monkeypatch.setattr(llm, "get_llm", lambda role: client)
    monkeypatch.setattr(prompt_registry, "get_prompt", lambda *args: prompt[0])
    monkeypatch.setattr(prompt_registry, "resolve_version", lambda *args: "v2")
    config, first, pinned = versions.generation_context()
    assert config["base_url"] == "https://gateway.test/v1"
    assert "secret" not in json.dumps(config)
    if changed == "prompt":
        prompt[0] = "second prompt"
    elif changed == "implementation":
        original_read = versions.Path.read_bytes
        monkeypatch.setattr(versions.Path, "read_bytes", lambda path: original_read(path) + b"\n# changed postprocessing")
    else:
        setattr(client, changed, {"model": "two", "base_url": "https://other.test/v1",
                                  "temperature": 0.5, "max_tokens": 4096, "thinking": "disabled"}[changed])
    _, second, _ = versions.generation_context()
    assert first != second
    pinned.chat_structured_resilient("different prompt", "user", object)
    assert calls[0][0] == "first prompt"
