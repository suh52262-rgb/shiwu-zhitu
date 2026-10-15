from types import SimpleNamespace
import importlib
from app.agent.schemas import Diagnosis
from app.models.student import StudentState


def test_candidate_summary_keeps_original_text_and_judgement(monkeypatch):
    graph=importlib.import_module('app.agent.graph')
    monkeypatch.setenv('PROMPT_DIAGNOSIS_VERSION','v5')
    original='把“摩擦力与拉力相加”写成合力14 N'
    diagnosis=Diagnosis(issue_type='calculation',issue=original,concept='',severity='low',
                        recommended_action='核对力的方向',finding_kind='observed_error')
    class Client:
        def chat_structured_resilient(self,*args):return SimpleNamespace(value=diagnosis,degraded=False,reasons=[])
    monkeypatch.setattr(graph,'get_llm',lambda role:Client())
    state=StudentState(student_id='s',problem_id='p',current_answer='F=10+4=14 N',has_obvious_error=True)
    result=graph.diagnose(state)
    assert result['diagnosed_issue']=='把摩擦力与拉力相加写成合力14 N'
    assert result['issue_type']=='calculation' and result['severity']=='low'
    assert result['diagnosis_text_kind']=='model_summary_not_verbatim'
    change=result['safety_substitutions'][-1]
    assert change['original']==original and not change['semantic_validated']
    assert not result.get('degraded_reasons')
    assert diagnosis.issue==original
