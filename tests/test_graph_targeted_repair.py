from uuid import uuid4
import pytest
from app.models.candidate_graph import CandidateGraph
from app.models.problem import Problem
from app.services.graph_versions import atomic_json
from app.services.problem_repository import ProblemRepository
from app.services.graph_authoring import GraphAuthoring

def candidate(valid=False,conflict=False):
    return CandidateGraph(problem_id='p',nodes=[
        {'id':'S1','name':'受力分析','type':'modeling','description':'only F'},
        {'id':'S2','name':'建立定律','type':'key_reasoning','description':'F=ma'},
        {'id':'S3','name':'计算','type':'calculation','description':'a=4'},
        {'id':'S4','name':'结果','type':'result','description':'4向右'}],
        edges=[['S1','S2'],['S2','S3'],['S3','S4']],valid_paths=[['S1','S2','S3','S4']],
        node_evidence={n:['Q001' if valid else 'Q999'] for n in ['S1','S2','S3','S4']},
        quantity_coverage={'大小':['S3'],'方向':['S4']},open_questions=['原题与参考解矛盾'] if conflict else [])

class FakeClient:
    last_call=None
    def reset_last_call(self):pass

def setup(tmp_path,monkeypatch,*,fail_repair=False,conflict=False):
    from app.services import graph_authoring as module
    from app.services import llm
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls)>1 and fail_repair:raise RuntimeError('repair network failed')
        return candidate(valid=len(calls)>1,conflict=conflict)
    monkeypatch.setattr(module,'generate_candidate_graph',generate)
    monkeypatch.setattr(llm,'get_llm',lambda _:FakeClient())
    repo=ProblemRepository(tmp_path/'bank')
    atomic_json(repo.problem_path('p'),Problem(problem_id='p',statement='2kg受到8N向右合力',answer='4向右',metadata={'source':'reviewed_import'}).model_dump())
    return GraphAuthoring(tmp_path/'jobs',repo,generator=generate,start_worker=False),calls

def test_one_repair_is_kept_only_after_strict_improvement(tmp_path,monkeypatch):
    engine,calls=setup(tmp_path,monkeypatch)
    submitted=engine.submit('p',str(uuid4()),quantities_asked=['大小','方向'])
    engine.run(submitted['job_id'])
    job=engine.load(submitted['job_id'],'p')
    assert len(calls)==2 and job['status']=='needs_review'
    assert job['quality']['ok'] and job['automatic_repairs'][0]['accepted'] is True
    assert calls[1]['correction']['quality_findings']
    assert job['automatic_repairs'][0]['before_quality']['ok'] is False
    assert job['review'] is None and job['publication'] is None

def test_failed_repair_preserves_initial_candidate_without_hiding_problem(tmp_path,monkeypatch):
    engine,calls=setup(tmp_path,monkeypatch,fail_repair=True)
    submitted=engine.submit('p',str(uuid4()))
    engine.run(submitted['job_id'])
    job=engine.load(submitted['job_id'],'p')
    assert len(calls)==2 and job['status']=='needs_changes'
    assert job['graph']['nodes'] and 'repair network failed' in job['automatic_repairs'][0]['error']
    assert job['automatic_repairs'][0]['accepted'] is False

@pytest.mark.parametrize('conflict,enabled',[(True,True),(False,False)])
def test_no_automatic_paid_repair_for_input_conflict_or_disabled_option(tmp_path,monkeypatch,conflict,enabled):
    engine,calls=setup(tmp_path,monkeypatch,conflict=conflict)
    submitted=engine.submit('p',str(uuid4()),repair_once=enabled)
    engine.run(submitted['job_id'])
    assert len(calls)==1 and engine.load(submitted['job_id'],'p')['automatic_repairs']==[]
