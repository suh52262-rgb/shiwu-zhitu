"""AI-reviewed graph-enabled development probes through actual API; never publish."""
import argparse,hashlib,json,os,sys,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from pydantic import BaseModel
from run_t4_batch import capture_requests,write,digest


class SampleReview(BaseModel):
    verdict:str
    case_ids:list[str]
    issues:list[str]


class OutputReview(BaseModel):
    serious_false_accusation:bool
    unsupported_cognition:bool
    serious_physics:bool
    invalid_quote:bool
    explanation:str
    quoted_output:list[str]


def strings(value):
    if isinstance(value,str):return [value]
    if isinstance(value,dict):return sum((strings(v) for v in value.values()),[])
    if isinstance(value,list):return sum((strings(v) for v in value),[])
    return []


def review_request(student_id,problem_id,attempt_id,run_id):
    from app.api.video import VideoReviewRequest
    return VideoReviewRequest(student_id=student_id,problem_id=problem_id,
                              attempt_id=attempt_id,run_id=run_id).model_dump()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--review-samples',action='store_true');p.add_argument('--sample-reviews',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);data=json.loads(a.dataset.read_text(encoding='utf-8'));case_ids={c['id'] for q in data['problems'] for c in q['cases']}
    from app.services.llm.factory import get_llm
    if a.review_samples:
        rule='独立审核AI工程题目、物理参考解、Graph路径与每条expected_finished/definite_error标注是否成立。题目明确要求说明计算依据；漏答一个目标不能完成；两条合法方法只需一条；带符号速度与速率应区分。不得将AI审核冒充教师确认。每条case_id必须列出，存在实质硬错则reject并明确issues。'
        def sample(name):
            result=get_llm('review').chat_structured(rule,json.dumps(data,ensure_ascii=False),SampleReview).model_dump()
            assert set(result['case_ids'])==case_ids and len(result['case_ids'])==len(case_ids)
            write(a.out/(name+'.json'),{'dataset_sha256':digest(a.dataset),'provenance':'independent_ai_review','reviewer_id':name,**result})
            print('SAMPLE',name,result['verdict'],flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(sample,('A','B')))
        return
    assert a.sample_reviews
    reviews=[json.loads((a.sample_reviews/(n+'.json')).read_text(encoding='utf-8')) for n in ('A','B')]
    assert all(r['dataset_sha256']==digest(a.dataset) and r['verdict']=='accept' and not r['issues'] for r in reviews)
    source={str(f.relative_to(ROOT)):digest(f) for base in (ROOT/'backend/app',ROOT/'prompts') for f in base.rglob('*') if f.suffix in ('.py','.md')}
    source[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    rule=('独立评实际学生收到的教学及复盘。允许合法替代方法及坐标约定；禁止从结果猜认知、实际路径；逐字引用必须来自所称来源。'
          '四项严重flag分别为严重误指控、无依据认知断言、严重物理矛盾和引用不实。教学hint与核对时重建分开。'
          'true必须逐字引用本输出，不能把题干/学生原文/Graph文字当受评输出；引文支持理由。')
    write(a.out/'protocol.json',{'dataset_sha256':digest(a.dataset),'sample_reviews':{n:digest(a.sample_reviews/(n+'.json')) for n in ('A','B')},
        'source':source,'gate':data['gate_predefined'],'review_rule':rule,'scope':data['scope'],'model':'deepseek-flash','prompts':{'diagnosis':'v5','tutor':'v5','review':'v4'}})
    for rel in source:
        target=a.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
    from accept_real_videos import disposable_services
    records=[]
    with disposable_services():
        from app.config import get_settings
        s=get_settings()
        for key in ('postgres_host','postgres_user','postgres_password','postgres_db'):setattr(s,key,os.environ[key.upper()])
        s.postgres_port=int(os.environ['POSTGRES_PORT']);s.qdrant_url=os.environ['QDRANT_URL'];s.llm_provider='deepseek';s.llm_model='deepseek-flash'
        for role in ('step_matcher','analyze_step','diagnosis','tutor','review'):setattr(s,role+'_model','deepseek-flash')
        os.environ['GRAPH_RELEASE_POLICY']='legacy-demo';os.environ['PROBLEMS_DIR']=str(a.out/'problems')
        for q in data['problems']:
            write(a.out/'problems'/q['id']/'problem.json',{'problem_id':q['id'],'statement':q['question'],'metadata':{'origin':'ai_engineering_probe'},'knowledge_points':['牛顿第二定律','动能定理','匀变速运动','重力做功']})
            write(a.out/'problems'/q['id']/'graph.json',q['graph'])
        from app.db.postgres import get_engine
        from app.db.tables import Base
        Base.metadata.create_all(get_engine())
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api import video
        from app.api.attempts import AnalyzeRequest
        from app.api.video import VideoReviewRequest
        video.RUNS_DIR=a.out/'materials'
        from app.services.review_material import build_synthesized_payload,write_material
        from app.services.graph_versions import graph_hash
        from app.models.problem_graph import ProblemGraph
        with TestClient(app) as client:
            for q in data['problems']:
                for case in q['cases']:
                    row={'case_id':case['id'],'question':q['question'],'reference':q['reference'],'case':case,'teaching':[],'status':'failed'}
                    try:
                        get_llm('tutor').request_guard.before_call()
                        with capture_requests() as calls:
                            req={'student_id':'graph_'+case['id'],'problem_id':q['id'],'answer':case['writing'],'request_hint':True}
                            AnalyzeRequest.model_validate(req)
                            review_request(req['student_id'],q['id'],'preflight','0'*32)
                            r=client.post('/attempt/analyze',json=req);row['teaching'].append({'request':req,'http_status':r.status_code,'response':r.json()});r.raise_for_status()
                            aid=r.json()['attempt_id'];state=client.get('/attempt/'+aid).json();row['attempt']=state
                            # Existing actual API must select and bind this graph, not graphless.
                            assert r.json().get('graph_mode') is True or (r.json().get('graph_mode')=='graph')
                            assert r.json()['graph_version']==graph_hash(ProblemGraph.model_validate(q['graph']))
                            if case['continue_help']:
                                req2={**req,'attempt_id':aid,'answer':''};r2=client.post('/attempt/analyze',json=req2)
                                row['teaching'].append({'request':req2,'http_status':r2.status_code,'response':r2.json()});r2.raise_for_status()
                            payload=build_synthesized_payload(student_id=req['student_id'],problem_id=q['id'],attempt_id=aid,rounds=[{'round':1,'answer':case['writing']}])
                            run_id=uuid.uuid4().hex;row['run_id']=run_id
                            folder=a.out/'materials'/run_id;write_material(folder,payload)
                            review=client.post('/video/review',json=review_request(req['student_id'],q['id'],aid,run_id))
                            row['review_http_status']=review.status_code;row['review_response']=review.json()
                            row['model_requests']=calls
                            review.raise_for_status()
                        row['status']='succeeded'
                    except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:500]
                    write(a.out/(case['id']+'.json'),row);records.append(row);print('CASE',case['id'],row['status'],flush=True)
    def score(name):
        for row in records:
            packet={'question':row['question'],'reference':row['reference'],'student_writing':row['case']['writing'],'teaching':[t['response'] for t in row['teaching']],
                    'review':row.get('review_response',{}).get('review')}
            try:
                raw=get_llm('review').chat_structured(rule,json.dumps(packet,ensure_ascii=False),OutputReview).model_dump()
                corpus=strings({'teaching':packet['teaching'],'review':packet['review']})
                valid=all(q and any(q in t for t in corpus) for q in raw['quoted_output'])
                if any(raw[k] for k in ('serious_false_accusation','unsupported_cognition','serious_physics','invalid_quote')):valid=valid and bool(raw['quoted_output'])
                value={'raw':raw,'valid':valid}
            except Exception as exc:value={'valid':False,'error':type(exc).__name__+': '+str(exc)[:300]}
            write(a.out/(row['case_id']+'_'+name+'_review.json'),value);print('GRADE',name,row['case_id'],value['valid'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(score,('A','B')))
    drift=[rel for rel,h in source.items() if digest(ROOT/rel)!=h]
    write(a.out/'execution_summary.json',{'cases':len(records),'succeeded':sum(r['status']=='succeeded' for r in records),'source_drift':drift,'scope':data['scope'],'formal_gate_pending_semantic_integration':True})
if __name__=='__main__':main()
