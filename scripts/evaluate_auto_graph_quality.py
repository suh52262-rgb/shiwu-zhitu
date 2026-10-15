"""Frozen paired development probes. Same-family AI judge, never teacher accuracy.

Run baseline before modifying production; distinct output directory for each arm.
Labels and rubric are frozen first; raw output/errors/call usage are never dropped.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from pydantic import BaseModel,Field
from app.services.graph_generator import generate_candidate_graph
from app.services.graph_validator import validate_graph
from app.services.knowledge_vocabulary import vocabulary,validate as check_vocabulary
from app.services.llm import get_llm

RUBRIC='''你是解题Graph的审阅模型。评的是输出图，不是学生，也不对生成者身份评分。
依据给出的原题和预先冻结的正确核对答案，逐节点核对：数值/符号/方向/物理条件、所有小问是否覆盖、多个必须回答的小问是否被画成任选路径、是否凭空增加条件或不成立的替代方法。
参考答案可能故意含错误，核对答案才是本探针的真值。图中的common_errors是可能错误的例子，不是正确步骤，不可把其中错误数值拿来指控图本身；但不得把信息不足断言为学生已犯物理错误或心理问题。
每个问题必须指出具体节点/路径与原文，并说明与哪条题设或核对答案冲突；没有明确证据允许空列表。给missing_quantities时使用提供的目标量原名。
需核对问题数仅作AI工程读数，不代表真人编辑次数或工作时间。'''

class Review(BaseModel):
    serious_physics_errors:list[str]=Field(default_factory=list)
    missing_quantities:list[str]=Field(default_factory=list)
    false_alternative_paths:list[str]=Field(default_factory=list)
    unsupported_assumptions:list[str]=Field(default_factory=list)
    wording_or_redundancy:list[str]=Field(default_factory=list)
    reference_conflict_flagged:bool=False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',choices=['v2','v3'],required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--case',help='只运行指定ID，作为另存的补充诊断，不覆盖全量批')
    args=ap.parse_args()
    os.environ['PROMPT_GRAPH_GENERATOR_VERSION']=args.version
    source=ROOT/'evaluation/datasets/auto_graph_quality_dev_v1.json'
    dataset=json.loads(source.read_text(encoding='utf-8'))
    if args.case:
        dataset['cases']=[c for c in dataset['cases'] if c['id']==args.case]
        if not dataset['cases']:raise ValueError('未知案例ID')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'dataset.json').write_bytes(source.read_bytes())
    files=['backend/app/services/graph_generator.py','backend/app/models/candidate_graph.py','backend/app/services/graph_authoring.py',f'prompts/graph_generator/{args.version}.md']
    manifest={'version':args.version,'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'rubric':RUBRIC,'rubric_sha256':hashlib.sha256(RUBRIC.encode()).hexdigest(),
              'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
              'review_scope':'one independent call per output, same-family AI; development evidence only'}
    manifest['selected_cases']=[c['id'] for c in dataset['cases']]
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('OUT',out,flush=True)
    for case in dataset['cases']:
        record={'case':case,'status':'started'}
        start=time.monotonic()
        try:
            client=get_llm('graph_generator');client.reset_last_call()
            extra={}
            if args.version=='v3':
                extra={'quantities_asked':case['quantities']}
            candidate=generate_candidate_graph(problem_id=case['id'],problem_statement=case['statement'],
                reference_solution=case['reference'],knowledge_vocabulary=vocabulary(),llm=client,**extra)
            record['candidate']=candidate.model_dump(mode='json')
            record['generation_call']=client.last_call.as_dict() if client.last_call else None
            record['generation_sec']=round(time.monotonic()-start,3)
            graph=candidate.to_problem_graph()
            points=list(candidate.knowledge_points)+[k for n in graph.nodes for k in n.knowledge_points]
            record['vocabulary']=check_vocabulary(points)
            record['rules']=validate_graph(graph,use_llm=False,quantities_asked=case['quantities']).model_dump(mode='json')
            judge=get_llm('graph_validator');judge.reset_last_call()
            evidence={'question':case['statement'],'reference':case['reference'],'frozen_truth':case['truth'],
                      'quantities':case['quantities'],'graph':graph.model_dump(mode='json')}
            review=judge.chat_structured(RUBRIC,json.dumps(evidence,ensure_ascii=False),Review)
            record['review']=review.model_dump()
            record['review_call']=judge.last_call.as_dict() if judge.last_call else None
            record['status']='completed'
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (out/(case['id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(case['id'],record['status'],record.get('review',{}),flush=True)
    records=[json.loads((out/(c['id']+'.json')).read_text(encoding='utf-8')) for c in dataset['cases']]
    summary={'total':len(records),'completed':sum(r['status']=='completed' for r in records),
             'critical_issue_counts':{k:sum(len(r.get('review',{}).get(k,[])) for r in records) for k in
                 ['serious_physics_errors','missing_quantities','false_alternative_paths','unsupported_assumptions']},
             'vocabulary_checked':sum('vocabulary' in r for r in records),
             'vocabulary_failures':sum('vocabulary' in r and not r['vocabulary']['ok'] for r in records),
             'rules_checked':sum('rules' in r for r in records),
             'rule_invalid':sum('rules' in r and not r['rules']['valid'] for r in records),
             'reference_conflict_flagged':next((r.get('review',{}).get('reference_conflict_flagged') for r in records if r['case']['id']=='reference_conflict'),None),
             'claim_scope':'six development probes; no population accuracy or human time claim'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
