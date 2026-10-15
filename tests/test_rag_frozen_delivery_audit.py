"""Protect distinctions that the archived evaluator could not establish."""
import importlib.util
from pathlib import Path
import pytest
import json

spec=importlib.util.spec_from_file_location("audit_delivery",Path(__file__).parents[1]/"scripts/audit_rag_frozen_delivery.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def hit(bid,ctx=()):
    return {"block_id":bid,"context_blocks":[{"id":x} for x in ctx]}

def test_rank_four_context_is_not_truncation():
    assert audit.classify("g",[hit("a"),hit("b"),hit("c"),hit("d",["g"])],[])=="rank_excluded"

def test_context_in_selected_window_is_actual_exclusion():
    assert audit.classify("g",[hit("a",["g"])],[])=="actual_delivery_window_excluded"

def test_missing_identity_not_miss():
    assert audit.classify("g",[hit(None)],[])=="unverifiable_identity"

def test_finite_miss_never_claims_global_absence():
    assert audit.classify("g",[hit("a")],[])=="not_seen_in_returned_top_k"

def test_present_id_does_not_prove_quote_coverage():
    r=audit.quote_coverage({"block_id":"g","excerpt_verbatim":"required sentence"},[{"id":"g","text":"required"}])
    assert r["identity_present"] and not r["exact_quote_delivered"]

def test_quote_in_another_source_is_not_support():
    assert not audit.quote_coverage({"block_id":"g","excerpt_verbatim":"abc"},[{"id":"other","text":"abc"}])["exact_quote_delivered"]

def test_identity_mismatch_detected_even_with_valid_uuid():
    facts={"a":{"doc_id":"book","page":1,"text":"abc"}}
    assert audit.identity_issues([{"block_id":"a","source":"other","page":1,"text":"abc"}],[],facts)

def test_partial_delivery_can_verify_source_without_proving_full_content():
    facts={"a":{"doc_id":"book","page":1,"text":"abc def"}}
    assert not audit.identity_issues([],[{"id":"a","source":"book","page":1,"text":"abc"}],facts)

@pytest.mark.parametrize("cases",[[{"id":"a"}],[{"id":"a","query":"x"},{"id":"a","query":"y"}]])
def test_invalid_cases_fail(cases):
    with pytest.raises(ValueError):audit.validate_cases({"drafts":cases})


def test_freeze_drift_blocks_before_any_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"files": {"annotation.json": audit.digest(annotation)}}))
    assert audit.verify_freeze(freeze, annotation)["failures"] == []
    annotation.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen inputs changed"):
        audit.verify_freeze(freeze, annotation)


def test_unbound_annotation_cannot_borrow_a_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="annotation not bound"):
        audit.verify_freeze(freeze, annotation)
