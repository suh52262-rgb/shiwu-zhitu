import json
from uuid import uuid4

import pytest
from app.models.candidate_graph import CandidateGraph
from app.models.problem import Problem
from app.services.graph_authoring import GraphAuthoring,GraphJobBusy
from app.services.graph_versions import atomic_json,GraphVersions,graph_hash
from app.services.graph_validator import validate_graph
from app.services.problem_repository import ProblemRepository
from app.services.problem_import import ImportConflict


def candidate(pid='p'):
    return CandidateGraph(problem_id=pid,nodes=[
        {'id':'S1','name':'研究对象','type':'modeling','description':'2kg物体在光滑面上，水平向右力8N'},
        {'id':'S2','name':'牛顿第二定律','type':'key_reasoning','description':'水平合力等于F'},
        {'id':'S3','name':'计算加速度','type':'calculation','description':'a=8/2=4m/s²'},
        {'id':'S4','name':'给出结果','type':'result','description':'大小4m/s²，向右'}],
        edges=[['S1','S2'],['S2','S3'],['S3','S4']],valid_paths=[['S1','S2','S3','S4']])


@pytest.fixture
def service(tmp_path):
    repo=ProblemRepository(tmp_path/'bank')
    atomic_json(repo.problem_path('p'),Problem(problem_id='p',statement='2kg物体受向右8N力，求加速度大小和方向',
                                              answer='4m/s²向右',metadata={'source':'reviewed_import'}).model_dump())
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        return candidate()
    engine=GraphAuthoring(tmp_path/'jobs',repo,generator=generate,start_worker=False)
    engine.calls=calls
    return engine


def finished(service,**kwargs):
    job=service.submit('p',str(uuid4()),**kwargs)
    service.run(job['job_id'])
    return service.load(job['job_id'],'p')


@pytest.mark.parametrize('cannot_save_failure',[False,True])
def test_worker_failure_does_not_retry_or_stop_next_job(service,monkeypatch,cannot_save_failure):
    first=service.submit('p',str(uuid4()))
    second_id=uuid4().hex
    second=dict(first,job_id=second_id,client_request_id=str(uuid4()))
    service._save(second)
    service.pending.put(second_id)
    seen=[]
    def run(jid):
        seen.append(jid)
        if jid==first['job_id']:raise RuntimeError('simulated worker exception')
        record=service.load(jid,'p');record['status']='needs_review';service._save(record)
    monkeypatch.setattr(service,'run',run)
    original_save=service._save
    def save(record):
        if cannot_save_failure and record['status']=='failed':raise OSError('simulated disk failure')
        original_save(record)
    monkeypatch.setattr(service,'_save',save)
    original_get=service.pending.get
    def get():
        if service.pending.empty():raise StopIteration
        return original_get()
    monkeypatch.setattr(service.pending,'get',get)
    with pytest.raises(StopIteration):service._worker()
    assert seen==[first['job_id'],second_id]
    assert service.pending.unfinished_tasks==0
    assert service.load(second_id,'p')['status']=='needs_review'
    failed=service.load(first['job_id'],'p')
    assert failed['status']==('queued' if cannot_save_failure else 'failed')
    if not cannot_save_failure:assert failed['error_type']=='RuntimeError'
    assert service.calls==[]


def test_generate_is_idempotent_durable_and_not_a_teaching_graph(service):
    rid=str(uuid4())
    first=service.submit('p',rid,quantities_asked=['加速度大小','加速度方向'])
    assert service.submit('p',rid,quantities_asked=['加速度大小','加速度方向'])['job_id']==first['job_id']
    assert service.calls==[]
    service.run(first['job_id'])
    result=service.load(first['job_id'],'p')
    assert len(service.calls)==1 and result['status']=='needs_review'
    assert result['validation']['checks']['physics']=='skipped'
    assert result['graph']['metadata']['needs_review'] is True
    assert service.repo.get_teaching_graph('p') is None
    assert service.submit('p',rid,quantities_asked=['加速度大小','加速度方向'])['job_id']==first['job_id']
    assert len(service.calls)==1
    with pytest.raises(ImportConflict):service.submit('p',rid,quantities_asked=['不同目标'])


def test_ai_review_is_recorded_without_unlocking_publication(service):
    job=finished(service)
    result=service.review('p',job['job_id'],job['graph_version'],'AI fixture','Checked fixture','ai',True)
    assert result['reviews'][0]['kind']=='ai' and result['review'] is None
    assert GraphVersions(service.repo.problem_dir('p')).manifest()['reviews']=={}
    with pytest.raises(ImportConflict):service.publish('p',job['job_id'],job['graph_version'],'AI fixture')


def test_human_fixture_review_publish_then_regenerate_keeps_old_teaching_snapshot(service):
    first=finished(service)
    service.review('p',first['job_id'],first['graph_version'],'human test fixture','All questions and paths checked','human',True)
    service.publish('p',first['job_id'],first['graph_version'],'human test fixture')
    old_hash=service.repo.graph_release('p')['graph_version']
    next_job=finished(service)
    assert next_job['graph_version']!=old_hash
    assert service.repo.graph_release('p')['graph_version']==old_hash
    assert service.repo.get_teaching_graph('p').metadata['authoring_job']==first['job_id']
    assert len(GraphVersions(service.repo.problem_dir('p')).manifest()['releases'])==1


def test_edit_invalidates_review_and_rejects_stale_version(service):
    job=finished(service)
    service.review('p',job['job_id'],job['graph_version'],'human fixture','Checked','human',True)
    changed=json.loads(json.dumps(job['graph']))
    changed['nodes'][-1]['description']='大小4m/s²，沿向右方向'
    changed['metadata']={'needs_review':False,'reviewed_by':'forged'}
    result=service.edit('p',job['job_id'],job['graph_version'],changed)
    assert result['review'] is None and result['graph']['metadata']['needs_review'] is True
    assert 'reviewed_by' not in result['graph']['metadata']
    with pytest.raises(ImportConflict):service.publish('p',job['job_id'],job['graph_version'],'human fixture')
    with pytest.raises(ImportConflict):service.publish('p',job['job_id'],result['graph_version'],'human fixture')


def test_invalid_edit_is_preserved_and_can_be_fixed_without_model_call(service):
    job=finished(service)
    bad=json.loads(json.dumps(job['graph']))
    bad['edges'].append(['S4','S1'])
    result=service.edit('p',job['job_id'],job['graph_version'],bad)
    assert result['status']=='needs_changes' and result['validation']['valid'] is False
    with pytest.raises(ValueError):service.review('p',job['job_id'],result['graph_version'],'human fixture','Checked','human',True)
    result=service.edit('p',job['job_id'],result['graph_version'],job['graph'])
    assert result['status']=='needs_review' and len(service.calls)==1


def test_restart_marks_queued_jobs_interrupted_and_never_reruns(service):
    job=service.submit('p',str(uuid4()))
    replacement=GraphAuthoring(service.root,service.repo,generator=service.generator,start_worker=False)
    assert replacement.load(job['job_id'],'p')['status']=='interrupted'
    assert service.calls==[]


def test_failure_can_be_found_and_retry_requires_new_request(service):
    def fail(**kwargs):raise RuntimeError('network failure')
    service.generator=fail
    rid=str(uuid4())
    job=service.submit('p',rid)
    service.run(job['job_id'])
    result=service.load(job['job_id'],'p')
    assert result['status']=='failed' and 'network failure' in result['error']
    assert service.submit('p',rid)['status']=='failed'
    assert service.submit('p',str(uuid4()))['status']=='queued'


def test_problem_change_during_generation_keeps_output_without_installing(service):
    def changed(**kwargs):
        problem=service.repo.get_problem('p')
        problem.statement+=' altered'
        atomic_json(service.repo.problem_path('p'),problem.model_dump())
        return candidate()
    service.generator=changed
    job=finished(service)
    assert job['status']=='stale' and job['graph']
    assert service.repo.get_graph('p') is None


def test_figure_only_paths_cannot_supply_missing_conditions(service):
    path=service.repo.problem_dir('p')/'figures/question.png'
    path.parent.mkdir()
    from PIL import Image
    Image.new('RGB',(20,20)).save(path)
    problem=service.repo.get_problem('p')
    problem.figures=['figures/question.png']
    atomic_json(service.repo.problem_path('p'),problem.model_dump())
    with pytest.raises(ValueError,match='图示说明'):service.submit('p',str(uuid4()))
    job=finished(service,figure_context='图示为2kg物块，光滑水平面，水平向右作用8N力')
    assert '操作者核对的图示说明' in service.calls[0]['problem_statement']
    assert len(job['input']['figures'])==1


def test_duplicate_running_job_and_queue_bound_are_visible(service):
    service.submit('p',str(uuid4()))
    with pytest.raises(GraphJobBusy):service.submit('p',str(uuid4()))
    for pid in ['q','r','s']:
        atomic_json(service.repo.problem_path(pid),Problem(problem_id=pid,statement='question',metadata={'source':'reviewed_import'}).model_dump())
    service.submit('q',str(uuid4()))
    service.submit('r',str(uuid4()))
    with pytest.raises(GraphJobBusy):service.submit('s',str(uuid4()))


def test_external_candidate_replacement_blocks_old_job_review(service):
    job=finished(service)
    other=candidate().to_problem_graph()
    service.repo.save_graph('p',other)
    with pytest.raises(ImportConflict):service.review('p',job['job_id'],job['graph_version'],'human fixture','Checked','human',True)


def test_physics_check_receives_enriched_problem_and_does_not_imply_human_review(service):
    seen=[]
    def validator(graph,**kwargs):
        seen.append(kwargs)
        return validate_graph(graph,problem=kwargs['problem'],use_llm=False,quantities_asked=kwargs['quantities_asked'])
    service.validator=validator
    job=finished(service,check_physics=True,figure_context='Explicit checked diagram description')
    assert seen[0]['use_llm'] is True and 'Explicit checked' in seen[0]['problem'].statement
    assert job['review'] is None

def test_unknown_generated_concepts_require_edit_and_cannot_be_published(service):
    def generate(**kwargs):
        result=candidate()
        result.nodes[0].knowledge_points=['not an approved concept']
        return result
    service.generator=generate
    job=finished(service)
    assert not job['vocabulary']['ok']
    with pytest.raises(ValueError,match='表外'):
        service.review('p',job['job_id'],job['graph_version'],'human fixture','Checked','human',True)
    graph=job['graph']
    graph['nodes'][0]['knowledge_points']=['牛顿第二定律']
    edited=service.edit('p',job['job_id'],job['graph_version'],graph)
    assert edited['vocabulary']['ok']
    assert edited['graph']['metadata']['knowledge_points']==['牛顿第二定律']

def test_publish_response_interruption_is_recoverable_without_duplicate_release(service,monkeypatch):
    job=finished(service)
    service.review('p',job['job_id'],job['graph_version'],'human fixture','Checked','human',True)
    original=service._save
    def fail(record):raise OSError('response interrupted')
    monkeypatch.setattr(service,'_save',fail)
    with pytest.raises(OSError):service.publish('p',job['job_id'],job['graph_version'],'human fixture')
    monkeypatch.setattr(service,'_save',original)
    result=service.publish('p',job['job_id'],job['graph_version'],'human fixture')
    assert result['publication'] and len(GraphVersions(service.repo.problem_dir('p')).manifest()['releases'])==1

def test_generator_vocabulary_constraint_is_explicit_and_does_not_invent_student_errors():
    from app.services.graph_generator import build_user_prompt
    prompt=build_user_prompt('question',knowledge_vocabulary=['牛顿第二定律'])
    assert '只能从以下列表选取' in prompt and '牛顿第二定律' in prompt
    assert '不要把信息不足当成已犯错误' in prompt
