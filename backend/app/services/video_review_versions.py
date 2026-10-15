"""Immutable video review materials and generation identity (never stores credentials)."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def timeline_items(state) -> list[dict]:
    return [event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
            for event in (getattr(state, "timeline", []) or [])]


def generation_context():
    """Pin the actual client and prompt used by this request, including cached prompts."""
    from app.agent.prompt_registry import get_prompt, resolve_version
    from app.services.llm import get_llm
    from app.services.review_builder import ReviewReport, build_review_user

    client = get_llm("review")
    prompt_version = resolve_version("review")
    prompt = get_prompt("review", prompt_version)
    url = urlsplit(client.base_url)
    # Gateways occasionally embed credentials in userinfo or a query string.
    safe_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    config = {"provider": client.name, "kind": client.kind, "model": client.model,
              "base_url": safe_url, "temperature": client.temperature,
              "timeout": client.timeout, "max_retries": client.max_retries,
              "max_tokens": getattr(client, "max_tokens", None),
                "thinking": getattr(client, "thinking", ""),
              "prompt_version": prompt_version,
              "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
              "schema_sha256": fingerprint(ReviewReport.model_json_schema()),
              "user_builder_sha256": hashlib.sha256(inspect.getsource(build_review_user).encode("utf-8")).hexdigest(),
              "review_implementation_sha256": hashlib.sha256(Path(inspect.getfile(build_review_user)).read_bytes()).hexdigest(),
              "review_contract": 1}

    class PinnedClient:
        def chat_structured_resilient(self, _system, user, schema):
            return client.chat_structured_resilient(prompt, user, schema)

    return config, fingerprint(config), PinnedClient()


def refresh_material(original: dict, current: dict, state) -> dict:
    """Allow only audited corrections to this run, never subsequent independent input."""
    if not isinstance(original.get("timeline_snapshot"), list) or not isinstance(original.get("media_source_ids"), list):
        raise HTTPException(409, "旧视频缺少可验证完整历史，请重新上传")
    baseline = original["timeline_length"]
    timeline = timeline_items(state)
    prefix = original.get("timeline_snapshot")
    if len(timeline) < baseline or (prefix is not None and timeline[:baseline] != prefix):
        raise HTTPException(409, "原视频作答历史已变化，不能更新复盘材料")
    sources = list(getattr(state, "media_sources", []) or [])
    source_ids = [str(item.get("source_id")) for item in sources]
    if "media_source_ids" in original and source_ids != original["media_source_ids"]:
        raise HTTPException(409, "该视频后已有独立作答，不能更新复盘材料")
    by_id = {str(item.get("source_id")): item for item in sources}
    expected = dict(original["source_revisions"])
    for event in timeline[baseline:]:
        match = re.fullmatch(r"source=([^;]+); revision=(\d+); supersedes=(\d+)",
                             event.get("content", ""))
        if event.get("event_type") != "ocr_correction" or not match:
            raise HTTPException(409, "该视频后已有独立作答，不能更新复盘材料")
        key, revision, before = match.group(1), int(match.group(2)), int(match.group(3))
        if key not in expected or expected[key] != before or revision != before + 1:
            raise HTTPException(409, "更正记录与视频来源不匹配")
        expected[key] = revision
    if any(by_id.get(key, {}).get("revision") != value for key, value in expected.items()):
        raise HTTPException(409, "更正记录与视频来源版本不匹配")
    if not expected or any(not r.get("source_id") for r in original["rounds"]):
        raise HTTPException(409, "旧视频缺少可验证来源，请重新上传")
    data = copy.deepcopy(original)
    for index, item in enumerate(data["rounds"]):
        key = item["source_id"]
        revision = expected.get(key)
        source = by_id.get(key, {})
        if revision == original["source_revisions"].get(key):
            continue
        history = source.get("revisions", [])
        latest = next((v for v in history if v.get("revision") == revision), None)
        if not latest or not isinstance(latest.get("text"), str) or not isinstance(latest.get("analysis"), dict):
            raise HTTPException(409, "更正来源缺少完整分析")
        analysis = latest["analysis"]
        if analysis.get("source_id") != key or analysis.get("source_revision") != revision:
            raise HTTPException(409, "更正分析与来源版本不匹配")
        # Replace the whole projection, retaining only immutable video coordinates.
        data["rounds"][index] = {**analysis, "round": item.get("round"), "ts": item.get("ts"),
                                "answer": latest["text"]}
    data["source_revisions"] = expected
    data["timeline_length"] = len(timeline)
    data["timeline_snapshot"] = timeline
    if data == current:
        return current
    return data
