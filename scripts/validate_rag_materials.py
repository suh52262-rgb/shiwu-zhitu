"""Frozen new retrieval probes, independently judged in two fresh model calls.

Anonymous packets hide arm identity, not the question or scoring rules. Both
judges are same-family AI calls and never see implementation or prior ratings.
Original data, material catalogues, errors, calls and disagreements are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from pydantic import BaseModel, Field

RUBRIC = '''只评提供的教材片段是否足以回答这个具体问题的全部必需要点，不评主题相似。
不要使用你自己的物理知识补足缺失材料。片段内有结论但无所问理由，仍不充分。
逐要点给covered/partial/missing，并引片段中真实逐字原文；没有依据不要编引文。
比较匿名两组X/Y，每组独立判定sufficient；不知道生成版本，不猜测。
评分对象是实际发送材料，不是检索能力、学生认知或代码正确性。允许两组都不充分。'''


class Assessment(BaseModel):
    sufficient: bool
    point_status: list[str]
    supporting_quotes: list[str] = Field(default_factory=list)
    explanation: str


class Pair(BaseModel):
    X: Assessment
    Y: Assessment


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(RAG_RERANK='0', RAG_QUERY_EXPANSION='0', EMBEDDING_PROVIDER='bge', HF_HUB_OFFLINE='1')
    from app.agent.tools import retrieve_knowledge
    from app.services.rag_materials import fragments
    from app.config import get_settings
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    dataset=ROOT/'evaluation/datasets/rag_material_validation_v1_20261013.json'
    data=json.loads(dataset.read_text(encoding='utf-8'))
    paths=[dataset, Path(__file__), ROOT/'backend/app/services/rag_materials.py', ROOT/'backend/app/rag/retriever.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write(args.out/'manifest.json', {'files':hashes,'criterion':data['acceptance'],'rubric':RUBRIC,
          'scope':'two independent same-family AI calls per new probe; not formal holdout',
          'retrieval_overrides':{'rerank':False,'query_expansion':False},'review_model':'deepseek-flash'})
    (args.out/'dataset.json').write_bytes(dataset.read_bytes())
    records=[]
    for index, case in enumerate(data['cases']):
        record={'case':case,'status':'started','reviews':[]}
        try:
            started=time.perf_counter()
            hits=retrieve_knowledge(case['query'],top_k=5)
            plans={p:fragments(hits,selected_policy=p) for p in ('legacy','hit_first_total_v1')}
            record.update(hits=hits,plans=plans,retrieval_sec=time.perf_counter()-started)
            # Balanced anonymous arm order, fixed before calls.
            mapping={'X':'legacy','Y':'hit_first_total_v1'} if index%2==0 else {'X':'hit_first_total_v1','Y':'legacy'}
            packet={'query':case['query'],'required_points':case['required_points'],
                    'materials':{key:plans[value]['sources'] for key,value in mapping.items()}}
            record['anonymous_mapping']=mapping
            for reviewer in ('A','B'):
                settings=get_settings()
                judge=OpenAICompatibleProvider(api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,model='deepseek-flash',
                    temperature=0.2,timeout=120,max_retries=1,name='deepseek')
                judge.reset_last_call()
                try:
                    review=judge.chat_structured(RUBRIC,json.dumps(packet,ensure_ascii=False),Pair).model_dump()
                except Exception:
                    record['reviews'].append({'reviewer':reviewer,'status':'failed',
                        'call':judge.last_call.as_dict() if judge.last_call else None})
                    write(args.out/(case['id']+'.json'),record)
                    raise
                valid=True
                for arm, result in review.items():
                    texts=[s['text'] for s in packet['materials'][arm]]
                    valid &= len(result['point_status'])==len(case['required_points'])
                    valid &= all(v in {'covered','partial','missing'} for v in result['point_status'])
                    valid &= all(any(q in t for t in texts) for q in result['supporting_quotes'])
                    valid &= not result['sufficient'] or all(v=='covered' for v in result['point_status'])
                record['reviews'].append({'reviewer':reviewer,'output':review,'valid':valid,
                       'call':judge.last_call.as_dict() if judge.last_call else None})
                write(args.out/(case['id']+'.json'),record)
            record['status']='completed'
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write(args.out/(case['id']+'.json'),record)
        records.append(record)
        print(case['id'], record['status'],flush=True)
    drift=[str(p.relative_to(ROOT)) for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p.relative_to(ROOT))]]
    gains,regressions,invalid=0,0,0
    for r in records:
        if r['status']!='completed':
            invalid+=1
            continue
        mapping=r['anonymous_mapping']
        outcomes=[]
        for review in r['reviews']:
            invalid+=not review['valid']
            values={mapping[k]:v['sufficient'] for k,v in review['output'].items()}
            regressions+=values['legacy'] and not values['hit_first_total_v1']
            outcomes.append(values['hit_first_total_v1'] and not values['legacy'])
        gains+=all(outcomes)
    accepted=gains>0 and regressions==0 and invalid==0 and not drift
    write(args.out/'summary.json',{'case_count':len(records),'completed':sum(r['status']=='completed' for r in records),
          'both_judges_confirmed_gains':gains,'individual_regression_judgements':regressions,
          'invalid_or_failed':invalid,'source_drift':drift,'candidate_accepted':accepted,
          'semantic_validated_by_humans':False,'decision_scope':'engineering probes, no population accuracy claim'})


if __name__=='__main__':
    main()
