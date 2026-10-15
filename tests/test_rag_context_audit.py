"""Production replay and independent audit failures; no network or paid models."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rag_context_audit", ROOT / "scripts/audit_rag_context_payload.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fact(identity, text, page=1, **kwargs):
    return {"id": identity, "doc_id": "doc", "page": page, "block_index": 0,
            "type": "text", "text": text, "source_kind": "text_layer",
            "indexable": True, "is_generated": False, "note": "", **kwargs}


@pytest.fixture
def material(monkeypatch):
    from app.rag import block_store
    blocks = [fact("bad", "不能发送的坏公式" * 60, indexable=False),
              fact("hit", "命中证据：必须依据实际书写核对这一段内容。"),
              fact("generated", "模型生成的解释不等于教材原文。", is_generated=True),
              fact("next", "下一页的实际内容应有自己的来源页码。", page=2)]
    for i, block in enumerate(blocks):
        block["block_index"] = i
    grouped = {("doc", page): [b for b in blocks if b["page"] == page] for page in (1, 2)}
    monkeypatch.setattr(block_store, "load_page_blocks", lambda doc, pages: deepcopy(grouped))
    facts = {b["id"]: b for b in blocks}
    row = {"id": "regression", "status": "succeeded", "hits": [{
        "source": "doc", "page": 1, "text": blocks[1]["text"],
        "context": "ARCHIVED_POISON", "context_blocks": [{"id": "bad"}]}]}
    return row, facts


@pytest.fixture
def render(monkeypatch):
    # This suite checks the historical per-hit policy; the total-budget catalogue
    # has its own production-input and provenance tests. No skipped cases.
    monkeypatch.setenv("RAG_MATERIAL_POLICY", "legacy")
    from app.agent import graph
    from app.models.student import StudentState
    capture = audit.PromptCapture()
    monkeypatch.setattr(graph, "get_llm", lambda role: capture)
    def call(hits):
        graph.generate_hint(StudentState(hint_level=1, current_answer="审计替身输入", retrieved_context=hits))
        return capture.messages.pop()
    return call


def test_replay_calls_current_production_and_discards_archived_expansion(material, render):
    from app.rag import retriever
    row, facts = material
    original = deepcopy(row)
    with patch.object(retriever, "_expand_context", wraps=retriever._expand_context) as production:
        result = audit.replay_case(row, facts, render, audit.detect_tutor_budget(render))
    assert production.call_count == 1
    assert production.call_args.kwargs == {"include_generated": False}
    assert row == original
    assert result["status"] == "verified"
    assert "ARCHIVED_POISON" not in result["captured_prompt"]["user"]
    check = result["checks"][0]
    assert check["hit"]["context"].startswith(row["hits"][0]["text"])
    assert [b["id"] for b in check["sent_fragments"]] == ["hit", "next"]
    assert check["flags"]["cross_page_material_in_tutor"]


def test_generated_opt_in_reaches_production_and_is_still_labelled(material, render):
    row, facts = material
    result = audit.replay_case(row, facts, render, 400, include_generated=True)
    assert result["status"] == "verified"
    assert result["checks"][0]["flags"]["generated_material_in_tutor"]
    assert not result["checks"][0]["flags"]["unindexable_material_in_tutor"]


def test_database_fallback_cannot_reuse_old_material_or_pass(material, render, monkeypatch):
    from app.rag import block_store
    def unavailable(*a, **kw):
        raise OSError("fake PG failure")
    monkeypatch.setattr(block_store, "load_page_blocks", unavailable)
    row, facts = material
    result = audit.replay_case(row, facts, render, 400)
    assert result["status"] == "failed"
    assert "missing_context_blocks" in result["checks"][0]["issues"]
    assert "ARCHIVED_POISON" not in result["captured_prompt"]["user"]


def test_audit_catches_regressed_filter_in_production(material, render, monkeypatch):
    from app.rag import retriever
    row, facts = material
    original = retriever._expand_context
    def broken(hits, **kwargs):
        original(hits, **kwargs)
        h = hits[0]
        bad = "不能发送的坏公式"
        facts["bad"]["text"] = bad
        start = len(h["context"]) + 1
        h["context"] += " " + bad
        h["context_blocks"].append({"id": "bad", "page": 1, "type": "text",
            "source": "doc", "block_index": 0, "source_kind": "text_layer",
            "context_start": start, "context_end": start+len(bad),
            "text_start": 0, "text_end": len(bad)})
    monkeypatch.setattr(retriever, "_expand_context", broken)
    result = audit.replay_case(row, facts, render, 400)
    assert result["status"] == "failed"
    assert "unindexable_material_sent" in result["checks"][0]["issues"]


def test_audit_catches_lost_hit_and_unknown_provenance(material, render, monkeypatch):
    from app.rag import retriever
    def broken(hits, **kwargs):
        hits[0]["context"] = "无来源的另一段材料。"
        hits[0]["context_blocks"] = [{"id": "nonexistent"}]
    monkeypatch.setattr(retriever, "_expand_context", broken)
    row, facts = material
    result = audit.replay_case(row, facts, render, 400)
    assert result["status"] == "failed"
    assert {"unknown_block_id", "hit_anchor_lost", "hit_not_attributed"} <= set(result["checks"][0]["issues"])


def test_source_directory_is_clipped_and_follows_rendering_order(monkeypatch, render):
    from app.rag import block_store
    hit = fact("hit", "有效命中。" * 130)
    blocks = [fact("before", "前邻块内容。" * 100), hit, fact("after", "后邻块内容。" * 100)]
    monkeypatch.setattr(block_store, "load_page_blocks", lambda *a: {("doc", 1): blocks})
    row = {"id": "long", "status": "succeeded", "hits": [{"source": "doc", "page": 1, "text": hit["text"]}]}
    result = audit.replay_case(row, {b["id"]: b for b in blocks}, render, 400)
    assert result["status"] == "verified"
    check = result["checks"][0]
    assert len(check["hit"]["context"]) == 600
    assert check["hit"]["context_blocks"] == [{
        "id": "hit", "page": 1, "source": "doc", "block_index": 0, "type": "text",
        "source_kind": "text_layer", "context_start": 0, "context_end": 600,
        "text_start": 0, "text_end": 600}]
    assert check["sent_fragments"][0]["text_end"] == 400


@pytest.mark.parametrize("field,value,issue", [
    ("page", 99, "source_metadata_mismatch:page"),
    ("context_end", 9999, "invalid_or_unordered_offsets"),
    ("text_end", 1, "source_text_mismatch"),
])
def test_forged_metadata_never_verifies(material, render, field, value, issue):
    row, facts = material
    result = audit.replay_case(row, facts, render, 400)
    hit = deepcopy(result["checks"][0]["hit"])
    hit["context_blocks"][0][field] = value
    assert issue in audit.inspect_context(hit, facts, 400)["issues"]


def test_renderer_failure_does_not_count_as_verified(material):
    row, facts = material
    result = audit.replay_case(row, facts, lambda hits: {"user": "没有参考材料"}, 400)
    assert result["status"] == "failed"
    assert "actual_tutor_payload_mismatch" in result["checks"][0]["issues"]


@pytest.mark.parametrize("defect", ["wrong_page", "missing", "extra", "reordered"])
def test_actual_fragment_inventory_detects_rendering_defects(material, render, defect):
    row, facts = material
    def broken(hits):
        prompt = render(hits)
        first = f"「{facts['hit']['text']}」（doc 第1页）"
        second = f"「{facts['next']['text']}」（doc 第2页）"
        assert first in prompt["user"] and second in prompt["user"]
        if defect == "wrong_page":
            prompt["user"] = prompt["user"].replace(second, second.replace("第2页", "第1页"))
        elif defect == "missing":
            prompt["user"] = prompt["user"].replace(second, "")
        elif defect == "extra":
            prompt["user"] = prompt["user"].replace(second, second + " 「未发送的材料」（doc 第9页）")
        else:
            prompt["user"] = prompt["user"].replace(first, "SWAP").replace(second, first).replace("SWAP", second)
        return prompt
    result = audit.replay_case(row, facts, broken, 400)
    assert result["status"] == "failed"
    assert "actual_tutor_payload_mismatch" in result["checks"][0]["issues"]


def test_empty_archive_case_is_unverified():
    result = audit.replay_case({"id": "empty", "status": "failed", "hits": []}, {}, None, 400)
    assert result["status"] == "unverified"


def test_budget_probe_uses_actual_tutor(render):
    assert audit.detect_tutor_budget(render) == 400
    with pytest.raises(ValueError, match="did not inject"):
        audit.detect_tutor_budget(lambda hits: {"user": "no context"})


def test_existing_output_is_refused_before_any_database_access(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path), "--out", str(tmp_path)])
    with pytest.raises(FileExistsError):
        audit.main()


def test_output_inside_archive_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path), "--out", str(tmp_path / "new")])
    with pytest.raises(ValueError, match="immutable"):
        audit.main()
