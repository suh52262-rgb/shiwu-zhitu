"""Question conflicts and result omissions are grounded in current input."""
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.models.student import StudentState
from app.services.subquestion_context import resolve_subquestion, render_missing_fields

ROOT=Path(__file__).resolve().parents[1]


def materials():
    folder=PROBLEMS/'demo_02'
    return (Problem.model_validate_json((folder/'problem.json').read_text(encoding='utf-8')),
            ProblemGraph.model_validate_json((folder/'graph.json').read_text(encoding='utf-8')))


@pytest.mark.parametrize('text,status', [
    ('(1) T = 0\nF=49N', 'conflict'),
    ('(3) T=0\nF=49N', 'resolved'),
    ('T=0\nF=49N', 'resolved'),
    ('(2) F=30N', 'resolved'),
])
def test_question_anchor_conflict_is_not_a_correctness_judgement(text,status):
    p,_=materials();c=resolve_subquestion(p,text)
    assert c['status']==status
    assert c['target']==('2' if '30N' in text else '3')


@pytest.mark.parametrize('text,direction', [
    ('(3) T=0\nF=49N',False), ('(3) T=0\nF=−49 N',False),
    ('(3) T=0\nF=49N，向右',True), ('(3) T=0\nF=49N，向左',True),
    ('(3) T=0\nF=m1a=−49N，向右',True),
    ('(3) T=0\n水平力大小为49N',False),
])
def test_magnitude_presence_is_separate_from_direction_and_sign(text,direction):
    p,_=materials();c=resolve_subquestion(p,text)
    fields={item['key']:item for item in c['result_fields']}
    assert fields['F_magnitude']['present']
    assert fields['F_direction']['present']==direction
    for item in fields.values():
        if item['present']:assert text[item['start']:item['end']]==item['quote']
    message,rejected=render_missing_fields(['F_magnitude','F_direction'],c)
    assert 'F_magnitude' in rejected and '大小' not in message
    assert ('方向' in message)==(not direction)


def test_conflict_skips_models_errors_progress_and_weak_concept_updates(monkeypatch):
    from app.agent import graph as G
    p,g=materials()
    monkeypatch.setattr(G,'get_llm',lambda *a:pytest.fail('Conflict must not call models'))
    result=G.tutor_graph.invoke(StudentState(current_answer='(1) T=0\nF=49N',hint_level=2,
        weak_concepts=['原有知识点'],visited_nodes=['S4']).model_dump(),
        config={'configurable':{'problem':p,'problem_graph':g}})
    assert result['subquestion_context']['status']=='conflict'
    assert result['diagnosed_issue']=='' and result['has_obvious_error'] is None
    assert result['weak_concepts']==['原有知识点'] and result['visited_nodes']==['S4']
    assert not result['finished'] and '暂不判定作答对错' in result['hint']


def test_reference_uses_signed_acceleration_and_distinguishes_slack_string():
    p,g=materials()
    # Numeric physical check under the reference's stated axes, independent of text rendering.
    m1,m2,gravity=5,2,9.8
    a=-gravity
    force=m1*a
    assert force == -49 and (force-m2*gravity)/(m1+m2)==pytest.approx(a)
    assert (m1+2*m2)*gravity != force
    assert 'a=−g' in p.answer and '松弛' in p.answer
    s6=g.node_map()['S6'].description
    assert 'a=−g' in s6
    assert '松弛后不可继续要求共同加速度' in s6


@pytest.mark.parametrize('text,error',[
    ('(3) T=0，F=49N，方向向左',True),
    ('(3) T=0，取F向左为正，F=-49N',False),
    ('(3) T=0，F=49N，不是向左，而是向右',False),
    ('(3) T=0，F=49N，是否向左？',False),
])
def test_direction_constraint_does_not_grade_axes_negation_or_questions(text,error):
    p,_=materials();c=resolve_subquestion(p,text)
    assert bool(c.get('verified_error'))==error


@pytest.mark.parametrize('text,valid',[
    ('(3) T=0。F=m1a，-m2g=m2a。',True),
    ('(3) T=0。m1a=F，m2a=-m2g。',True),
    ('(3) T=0。F=m1a，-m2g=2m2a。',False),
    ('(3) T=0。F=m1a，-m2g=m2a。机械能一定守恒',False),
])
def test_partial_equations_use_algebra_and_refuse_unknown_claims(text,valid):
    p,_=materials();c=resolve_subquestion(p,text)
    assert bool(c.get('verified_partial'))==valid
