"""Revision-aware OCR correction: sandbox before committing a replacement projection.

Source snapshots live in the existing state JSON; no schema migration is needed.
The process lock has the same single-worker limitation as ordinary analysis.
"""
from __future__ import annotations
import time
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.db import persistence
from app.models.attempt import AttemptRecord, STATUS_FINISHED
from app.models.student import TimelineEvent
from app.services.attempt_store import AttemptStore
from app.services.problem_binding import student_view
from app.services import attempt_store


class CorrectionRequest(BaseModel):
    student_id: str
    problem_id: str
    attempt_id: str
    source_id: str
    base_revision: int = Field(ge=0)
    corrected_text: str = Field(min_length=1, max_length=50000)


def finalize_media_profile(record: AttemptRecord) -> None:
    """Commit now-immutable media evidence before later input or lifecycle closure.

    Profile storage uses set union. Failed writes remain retryable; no media-derived
    weak concept is written before its correction window closes.
    """
    if not record.state:
        return
    if record.state.media_sources:
        latest = record.state.media_sources[-1]
        if not latest.get("correction_closed"):
            state = record.state
            persistence.resolve_pending_hints(
                record.attempt_id, visited_after=list(state.visited_nodes),
                current_node_after=state.current_node, correct_after=state.round_correct,
                consecutive_failures_after=state.consecutive_failures,
                off_graph_after=state.round_off_graph, degraded_after=bool(state.degraded_reasons),
                exclude_source_id=latest["source_id"],
            )
    for source in record.state.media_sources:
        source["correction_closed"] = True
        source.pop("before_record", None)
        if not source.get("profile_finalized") and source.get("revisions"):
            analysis = source["revisions"][-1]["analysis"]
            if persistence.persist_profile_weak_concepts(record.student_id, analysis.get("weak_concepts", [])):
                source["profile_finalized"] = True


def _response(record, source, *, idempotent=False):
    latest = source["revisions"][-1]
    return {
        "attempt_id": record.attempt_id, "source_id": source["source_id"],
        "source_revision": source["revision"], "original_text": source["original_text"],
        "corrected_text": latest["text"], "analysis": latest["analysis"],
        "state": student_view(record.state.model_dump(mode="json")),
        "timeline": [event.model_dump(mode="json") for event in record.state.timeline],
        "idempotent": idempotent,
    }


def correct_ocr(req: CorrectionRequest) -> dict:
    from app.api.attempts import AnalyzeRequest, _analyze_locked, _attempt_lock, _elapsed_seconds, validate_explicit_attempt

    text = req.corrected_text.strip()
    if not text:
        raise HTTPException(422, "更正文本不能为空")
    store = attempt_store.get_attempt_store()
    with _attempt_lock(store, req.student_id, req.problem_id):
        record = validate_explicit_attempt(req.attempt_id, req.student_id, req.problem_id)
        if record is None or record.state is None:
            raise HTTPException(404, "作答状态不存在")
        sources = record.state.media_sources
        source = next((item for item in sources if item["source_id"] == req.source_id), None)
        if source is None:
            raise HTTPException(404, "识别来源不存在")
        # Idempotency cannot bypass ownership, lifecycle, or later-input protection.
        if source.get("correction_closed") or source is not sources[-1] or len(record.state.timeline) != source["timeline_length"]:
            raise HTTPException(409, "该识别结果后已有新的作答，请勿修改历史分析")
        if req.base_revision != source["revision"]:
            latest = source["revisions"][-1]
            if (req.base_revision == latest.get("base_revision") and text == latest["text"]):
                return _response(record, source, idempotent=True)
            raise HTTPException(409, "识别版本已更新，请刷新后再更正")
        if text == source["revisions"][-1]["text"]:
            return _response(record, source, idempotent=True)

        before = AttemptRecord.model_validate(source["before_record"])
        sandbox = AttemptStore(persistent=False)
        sandbox._memory[record.attempt_id] = before.model_copy(deep=True)
        analyze_req = AnalyzeRequest.model_validate({
            **source["request"], "attempt_id": record.attempt_id,
            "answer": text, "media_type": "",
        })
        from app.agent.prompt_registry import prompt_version_fields
        collectors = []
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        before_state = record.state.model_dump(mode="json", exclude={"media_sources"})

        def trace_fields():
            usage = collectors[-1] if collectors else None
            return {"request_id": request_id, "problem_id": req.problem_id,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "llm_latency_ms": usage.latency_ms if usage else 0,
                    "provider": "+".join(usage.providers) if usage else "",
                    "model": usage.model_label() if usage else "",
                    "token_usage": usage.as_dict() if usage else {},
                    "prompt_versions": prompt_version_fields(),
                    "before_state": before_state}

        try:
            analysis = _analyze_locked(analyze_req, sandbox, record_override=before, sandbox=True,
                                       usage_capture=collectors)
            if analysis.degraded:
                raise HTTPException(503, "更正重算未完成，原分析保持不变")
        except Exception as exc:
            persistence.persist_trace(record.attempt_id,
                                      **trace_fields(),
                                      round_index=sum(e.event_type == "submit" for e in record.state.timeline),
                                      payload={"event_type": "ocr_correction_failed", "source_id": req.source_id,
                                               "base_revision": req.base_revision, "student_input": text,
                                               "error": str(exc)[:500], "applied": False})
            raise
        replacement = sandbox.load_attempt(record.attempt_id)
        revision = source["revision"] + 1
        analysis.source_id = req.source_id
        analysis.source_revision = revision
        source["revision"] = revision
        source["revisions"].append({
            "revision": revision, "base_revision": req.base_revision,
            "text": text, "analysis": analysis.model_dump(),
            "supersedes": {"source_id": req.source_id, "revision": req.base_revision},
        })
        audit = TimelineEvent(
            timestamp=_elapsed_seconds(record),
            event_type="ocr_correction",
            content=f"source={req.source_id}; revision={revision}; supersedes={req.base_revision}",
        )
        # Retain the actual history; only the current state projection is replaced.
        for evidence in replacement.state.error_evidence[len(before.state.error_evidence):]:
            evidence.update(source_id=req.source_id, source_revision=revision,
                            request_id=request_id)
        replacement.state.timeline = [*record.state.timeline, audit]
        replacement.state.media_sources = sources
        source["timeline_length"] = len(replacement.state.timeline)
        replacement.hint_history = [*before.hint_history]
        if analysis.hint:
            replacement.hint_history.append({"hint_level": analysis.hint_level, "hint": analysis.hint,
                                            "issue_type": analysis.issue_type, "source_id": req.source_id,
                                            "source_revision": revision})
        saved = store.save_attempt(replacement, replacement.state,
                                   error_count=replacement.error_count, hint_count=replacement.hint_count,
                                   stuck_count=replacement.stuck_count)
        persistence.supersede_media_hints(record.attempt_id, req.source_id, req.base_revision)
        if analysis.hint:
            persistence.persist_hint(record.attempt_id, analysis.hint_level, analysis.hint, analysis.issue_type,
                                     node=analysis.current_node,
                                     state_before={"source_id": req.source_id, "source_revision": revision,
                                                   "visited_nodes": list(replacement.state.visited_nodes),
                                                   "consecutive_failures": replacement.state.consecutive_failures})
        persistence.persist_events(record.attempt_id, [audit])
        persistence.persist_trace(record.attempt_id,
                                  **trace_fields(),
                                  round_index=sum(e.event_type == "submit" for e in saved.state.timeline),
                                  payload={"event_type": "ocr_correction", "source_id": req.source_id,
                                           "source_revision": revision,
                                           "graph_version": saved.state.graph_version,
                                           "graph_binding_status": saved.state.graph_binding_status,
                                           "supersedes": {"source_id": req.source_id, "revision": req.base_revision},
                                           "student_input": text, "analysis": analysis.model_dump()})
        if analysis.finished:
            finalize_media_profile(saved)
            store.save_attempt(saved, saved.state)
            store.finish_attempt(record.attempt_id, final_correct=True, status=STATUS_FINISHED)
            saved = store.load_attempt(record.attempt_id)
        return _response(saved, source)
