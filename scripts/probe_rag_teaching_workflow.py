"""Real retrieval -> teaching -> review, read-only corpus, no student DB writes.

This runs the production LangGraph and report builder, not the HTTP adapter.
Six exposed development topics are not a holdout. No tuning based on gold blocks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from pydantic import BaseModel


class Judgement(BaseModel):
    material_sufficient: bool
    hint_appropriate: bool
    hint_level_respected: bool
    review_appropriate: bool
    unsupported_cognitive_claim: bool
    invalid_source_quote: bool
    serious_physics_contradiction: bool
    severe_false_accusation: bool
    explanation: str
    quoted_output: list[str]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def strings(value):
    if isinstance(value,str):return [value]
    if isinstance(value,dict):return sum((strings(v) for v in value.values()),[])
    if isinstance(value,list):return sum((strings(v) for v in value),[])
    return []


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Use a fresh directory; preserve all originals')
    args.out.mkdir(parents=True)
    dataset=ROOT/'evaluation/datasets/rag_teaching_workflow_dev_20261013.json'
    data=json.loads(dataset.read_text(encoding='utf-8'))
    from app.config import get_settings
    settings=get_settings();settings.llm_provider='deepseek';settings.llm_model='deepseek-flash'
    for role in ('step_matcher','analyze_step','diagnosis','tutor','review'):
        setattr(settings,role+'_model','deepseek-flash')
    from sqlalchemy import event
    from app.db.postgres import get_engine
    engine=get_engine()
    def read_only(connection):connection.exec_driver_sql('SET TRANSACTION READ ONLY')
    event.listen(engine,'begin',read_only)
    from app.db.qdrant import get_qdrant
    qdrant=get_qdrant()
    initial_count=qdrant.count(settings.qdrant_collection,exact=True).count
    from app.agent.graph import tutor_graph, delivered_rag_sources
    from app.services.runtime_configuration import public_runtime_configuration
    from app.services.review_builder import review_trajectory
    from app.services.llm.factory import get_llm
    from app.services.llm.tracking import track_usage
    from app.models.student import StudentState,TimelineEvent
    from app.models.problem import Problem
    identity={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for base in (ROOT/'backend/app',ROOT/'prompts') for p in base.rglob('*') if p.suffix in {'.py','.md'}}
    for rel,h in identity.items():
        path=args.out/'source_snapshot'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/rel).read_bytes())
    rule=('独立评当前提示和复盘。你只收到本例题目、预先定义要点、纸面、实际加入提示请求的教材片段和输出。'
          '教材充分性与提示正确性分开；不能从主题相关推断机制覆盖。'
          'L1方向、L2概念、L3方法不直接列完整公式终值，L4可关键式但不完整答案；不得降低学生的当前等级。'
          '完整正确作答的进一步说明明确为可选时，不当作学生缺失步骤。'
          '禁止心智或未展示路径断言；只指出纸面关系、运算和结论。引用须在所称来源逐字存在。'
          '核对时重建不冒充学生实际过程。reported validation仅结构引用，不证明物理语义。'
          '严重误指控指把合法正确关系明确判为错误；与措辞不够精确区别。'
          '提示等级与泄题限制只评teaching hint，不套用于诊断issue或之后的完整复盘核对。'
          'quoted_output只能逐字摘本提示、issue或复盘，不能摘教材、题干、学生原文或评分说明冒充受评输出。严重flag为true必须给引用。')
    write(args.out/'protocol.json',{'data':data,'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'source':identity,'configuration':public_runtime_configuration(),'rule':rule,
        'qdrant_points':initial_count,'SQL_corpus_read_only':True,
        'rerank_setting':os.getenv('RAG_RERANK','1'),'query_expansion_setting':os.getenv('RAG_QUERY_EXPANSION','0'),
        'protocol_version':'rag_workflow_probe.v2',
        'gate':{'nondegraded_teaching_min':5,'valid_reviews_per_stream':6,'appropriate_hints_min_per_stream':5,
                'four_severe_flags_each_max':0,'source_drift_max':0}})
    records=[]
    for case in data['cases']:
        problem=Problem(problem_id='rag_workflow_'+case['id'],statement=case['question'],
                        answer=case['reference'],knowledge_points=[case['concept']])
        state=StudentState(student_id='ai_workflow_probe',problem_id=problem.problem_id,
            problem_content=problem.statement,current_answer=case['writing'],request_hint=True,
            profile_policy_version='paper_evidence_v1',timeline=[TimelineEvent(event_type='submit',content=case['writing'])])
        started=time.perf_counter()
        with track_usage() as usage:
            result=tutor_graph.invoke(state.model_dump(),config={'configurable':{'problem':problem,'problem_graph':None}})
        record={'case':case,'teaching':result,'teaching_usage':usage.as_dict(),
                'teaching_elapsed_s':time.perf_counter()-started,'reviews':[]}
        write(args.out/f"{case['id']}.json",record)
        with track_usage() as review_usage:
            report=review_trajectory(problem,[{'round':1,'answer':case['writing'],**result}])
        record['review']=report.model_dump() if report else None;record['review_usage']=review_usage.as_dict()
        record['review_raw_generation']=getattr(report,'_raw_generation',None) if report else None
        sources=delivered_rag_sources(result.get('retrieved_context')) if result.get('rag_materials_in_prompt') else []
        record['prompt_materials']=sources
        packet={'question':case['question'],'reference':case['reference'],'required_points':case['points'],
                'student_writing':case['writing'],'hint':result.get('hint'),
                'hint_level':result.get('hint_level'),'issue':result.get('diagnosed_issue'),
                'degraded_reasons':result.get('degraded_reasons'), 'materials':sources,'review':record['review']}
        output_strings=strings({'hint':packet['hint'],'issue':packet['issue'],'review':record['review']})
        for stream in ('A','B'):
            with track_usage() as grader_usage:
                outcome=get_llm('review').chat_structured_resilient(rule,json.dumps(packet,ensure_ascii=False),Judgement)
            value=outcome.value.model_dump() if outcome.value else None
            valid=bool(value and not outcome.degraded and all(q and any(q in text for text in output_strings) for q in value['quoted_output']))
            if value and any(value[k] for k in ('unsupported_cognitive_claim','invalid_source_quote','serious_physics_contradiction','severe_false_accusation')):
                valid=valid and bool(value['quoted_output'])
            record['reviews'].append({'stream':stream,'value':value,'valid':valid,'reasons':outcome.reasons,'usage':grader_usage.as_dict()})
            write(args.out/f"{case['id']}.json",record)
        records.append(record)
        print('COMPLETE '+case['id']+' RAG '+str(result.get('rag_status')),flush=True)
    serious=('unsupported_cognitive_claim','invalid_source_quote','serious_physics_contradiction','severe_false_accusation')
    summaries={}
    for stream in ('A','B'):
        reviews=[next(r for r in x['reviews'] if r['stream']==stream) for x in records]
        summaries[stream]={'valid':sum(r['valid'] for r in reviews),
            'material_sufficient':sum(r['valid'] and r['value']['material_sufficient'] for r in reviews),
            'hint_appropriate':sum(r['valid'] and r['value']['hint_appropriate'] and r['value']['hint_level_respected'] for r in reviews),
            'severe':{key:sum(r['valid'] and r['value'][key] for r in reviews) for key in serious},
            'invalid_review_cases':[x['case']['id'] for x,r in zip(records,reviews) if not r['valid']]}
    nondegraded=sum(not x['teaching'].get('degraded_reasons') for x in records)
    drift=[rel for rel,h in identity.items() if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h]
    passed=nondegraded>=5 and not drift and all(s['valid']==6 and s['hint_appropriate']>=5 and not any(s['severe'].values()) for s in summaries.values())
    write(args.out/'summary.json',{'cases':len(records),'nondegraded_teaching':nondegraded,'streams':summaries,
        'source_drift':drift,'passed':passed,'qdrant_points_before':initial_count,
        'qdrant_points_after':qdrant.count(settings.qdrant_collection,exact=True).count,
        'scope':'production agent/review flow, real read-only corpus; excludes HTTP/persistence; AI development probes'})


if __name__=='__main__':main()
