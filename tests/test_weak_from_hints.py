"""Help intensity must not manufacture writing-error evidence."""
from types import SimpleNamespace
import pytest
from app.agent.graph import hint_policy, diagnose
from app.models.student import StudentState
from app.agent.schemas import Diagnosis
from app.services.learning_profile import profile_signals


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_correct_help_escalates_without_weakness(level):
    state = StudentState(request_hint=True, round_correct=True, hint_level=level,
                         current_node="S1", weak_concepts=["已有书面错误"])
    result = hint_policy(state)
    assert result["hint_level"] >= 2
    assert "weak_concepts" not in result
    assert result["help_events"][-1]["node"] == "S1"
    assert state.weak_concepts == ["已有书面错误"]


def test_automatic_high_hint_not_help_or_weakness():
    result = hint_policy(StudentState(consecutive_failures=4, hint_level=3))
    assert "weak_concepts" not in result
    assert "help_events" not in result


@pytest.mark.parametrize("kind", ["missing_information", "uncertain", "none"])
def test_non_error_diagnosis_cannot_create_profile_evidence(monkeypatch, kind):
    import app.agent.graph as module
    diag = Diagnosis(finding_kind=kind, concept="动能定理", issue="需要核对")
    monkeypatch.setattr(module, "get_llm", lambda _: SimpleNamespace(
        chat_structured_resilient=lambda *args: SimpleNamespace(value=diag, degraded=False)))
    result = diagnose(StudentState(has_obvious_error=True, current_answer="E=2"))
    assert not result.get("weak_concepts")
    assert not result.get("error_evidence")


def test_profile_keeps_history_out_of_recommendation_basis():
    result = profile_signals([("old", {"weak_concepts": ["仅求助"]}),
        ("new", {"profile_policy_version": "paper_evidence_v1", "help_events": [{"level": 5}],
        "error_evidence": [{"concept": "动能定理", "issue": "关系错误", "student_text": "E=mv"}]})])
    assert result["observed_error_concepts"] == ["动能定理"]
    assert result["legacy_weak_concepts"] == ["仅求助"]
    assert len(result["help_events"]) == 1
    assert result["error_evidence"][0]["attempt_id"] == "new"
    assert result["semantic_validated"] is False
