import json
from pathlib import Path
from uuid import uuid4
import httpx
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.services.graph_authoring import GraphAuthoring
from app.services.graph_versions import atomic_json
from app.services.problem_repository import ProblemRepository
from app.models.problem import Problem
from app.models.candidate_graph import CandidateGraph


def graph():
    return CandidateGraph(problem_id='p',nodes=[
        {'id':'S1','name':'分析受力','type':'modeling','description':'仅受水平力F'},
        {'id':'S2','name':'确定定律','type':'key_reasoning','description':'牛顿第二定律'},
        {'id':'S3','name':'列式计算','type':'calculation','description':'a=F/m'},
        {'id':'S4','name':'给出结果','type':'result','description':'4m/s²向右'}],
        edges=[['S1','S2'],['S2','S3'],['S3','S4']],valid_paths=[['S1','S2','S3','S4']])


def test_api_job_lookup_edit_review_publish_are_exact_version_bound(tmp_path,monkeypatch):
    from app.api import graph_authoring
    monkeypatch.setenv('PROBLEMS_DIR',str(tmp_path/'bank'))
    repo=ProblemRepository(tmp_path/'bank')
    atomic_json(repo.problem_path('p'),Problem(problem_id='p',statement='2kg受8N水平力，求a',metadata={'source':'reviewed_import'}).model_dump())
    engine=GraphAuthoring(tmp_path/'jobs',repo,generator=lambda **_:graph(),start_worker=False)
    monkeypatch.setattr(graph_authoring,'get_graph_authoring',lambda:engine)
    from app.main import app
    client=TestClient(app)
    base='/problems/p/graph-jobs'
    request={'client_request_id':str(uuid4())}
    job=client.post(base,json=request).json()
    assert job['status']=='queued'
    assert client.post(base,json=request).json()['job_id']==job['job_id']
    assert client.post(base,json={'client_request_id':str(uuid4())}).status_code==429
    engine.run(job['job_id'])
    path=base+'/'+job['job_id']
    job=client.get(path).json()
    assert job['status']=='needs_review'
    assert client.get('/problems/p/graph').status_code==404
    version=job['graph_version']
    assert client.post(path+'/review',json={'expected_version':'stale','reviewer':'operator','note':'checked','confirmed':True}).status_code==409
    assert client.post(path+'/publish',json={'expected_version':version,'publisher':'operator'}).status_code==409
    assert client.post(path+'/review',json={'expected_version':version,'reviewer':'human fixture','note':'All questions and physics checked','confirmed':True}).status_code==200
    assert client.post(path+'/publish',json={'expected_version':version,'publisher':'human fixture'}).status_code==200
    assert client.get('/problems/p/graph').json()['graph_version']==version
    assert client.get(base).json()['teaching_version']==version
    assert client.get('/problems/q/graph-jobs/'+job['job_id']).status_code==422


def test_import_ui_published_question_can_generate_and_inspect_unpublished_graph(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    item={'item_id':'item1','problem_id':'p','title':'导入题','statement':'2kg物体受8N力','answer':'4m/s²',
          'source_pages':[1],'crops':[],'revision':1,'review':{'reviewer':'fixture'},
          'publication':{'problem_id':'p'},'history':[],'original_statement':'original','original_answer':'original'}
    record={'import_id':'abc','filename':'image.png','status':'draft','source_sha256':'a'*64,'warnings':[],
            'pages':[{'page':1,'method':'vision','status':'ready','original_text':'original'}],'items':[item]}
    candidate=graph().to_problem_graph().model_dump(mode='json')
    job={'job_id':'j1','status':'needs_review','phase':'complete','graph_version':'a'*64,'input_sha256':'b'*64,
         'revision':0,'graph':candidate,'validation':{'valid':True,'checks':{'physics':'skipped'}},'error':'',
         'reviews':[],'review':None,'publication':None,'input':{'options':{'figure_context':'','quantities_asked':[],'check_physics':False}},
         'timings':{}}
    calls=[]
    generated=[]
    def request(method,url,**kwargs):
        if method=='POST' and url.endswith('/graph-jobs'):
            calls.append(kwargs['json']);generated.append(job);result=job
        elif url.endswith('/graph-jobs'):
            result={'items':generated,'teaching_version':''}
        elif url.endswith('/problem-imports'):
            result={'items':[record]}
        else:result=record
        return httpx.Response(200,json=result,request=httpx.Request(method,url))
    monkeypatch.setattr(httpx,'request',request)
    app=AppTest.from_file(str(root/'frontend/problem_import_app.py')).run()
    assert not app.exception
    next(b for b in app.button if b.label=='生成候选Graph（调用模型）').click().run()
    assert not app.exception and len(calls)==1
    assert calls[0]['check_physics'] is False
    assert next(b for b in app.button if b.label=='发布已人工复核的Graph').disabled
    assert any('不能证明解法正确' in warning.value for warning in app.warning)
    # Repeating submit reuses the same persisted request identity; API handles
    # idempotence, so a UI retry cannot silently create a second paid request.
    next(b for b in app.button if b.label=='生成候选Graph（调用模型）').click().run()
    assert calls[1]['client_request_id']==calls[0]['client_request_id']
