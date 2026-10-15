"""Two fresh sequential AI reviews of one actual delivery plan; no teaching claims."""
import argparse,json,os,sys
from pathlib import Path
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from probe_rag_prose_spans import sha,write
from review_rag_page_evidence import RULE,Assessment
from rag_bounded_model_call import SingleAttemptStructuredClient


class OCRIssue(BaseModel):
    arm:str
    evidence_ids:list[str]=Field(default_factory=list)
    explanation:str
    blocks_required_point:bool


class Review(BaseModel):
    X:Assessment
    Y:Assessment
    ocr_issues:list[OCRIssue]=Field(default_factory=list)


def validate_review(output,facts):
    for arm in ['X','Y']:
        points=output[arm]['points']
        if len(points)!=2:raise ValueError('Missing required point ratings')
        for point in points:
            ids=point['evidence_ids']
            if len(ids)!=len(set(ids)) or any(i not in facts[arm] for i in ids):raise ValueError('Invalid or cross-arm evidence')
            if point['status']=='covered' and not ids:raise ValueError('Covered without evidence')
            point['server_evidence']=[facts[arm][i] for i in ids]
        output[arm]['sufficient']=all(p['status']=='covered' for p in points)
    for issue in output['ocr_issues']:
        arm=issue['arm'];ids=issue['evidence_ids']
        if arm not in facts or not ids or len(ids)!=len(set(ids)) or any(i not in facts[arm] for i in ids):raise ValueError('Invalid OCR issue source')
        issue['server_evidence']=[facts[arm][i] for i in ids]
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reviewers',nargs='+',choices=['A','B'],default=['A','B'])
    p.add_argument('--request-timeout',type=float,default=60)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    if len(set(a.reviewers))!=len(a.reviewers) or a.request_timeout<=0:raise ValueError('Invalid reviewer/timeout bound')
    input_path=a.selection/'result.json';result=json.loads(input_path.read_text(encoding='utf-8'))
    dataset=ROOT/'evaluation/datasets/rag_region_queries_20261014.json'
    case=next(c for c in json.loads(dataset.read_text(encoding='utf-8'))['cases'] if c['id']==result['id'])
    if result['status']!='completed' or result['selected_plan']['selection_status']!='ready':raise ValueError('Selection not ready')
    plans={'baseline':result['baseline_plan'],'selected':result['selected_plan']}
    if any(sum(len(s['text']) for s in plan['sources'])>1200 for plan in plans.values()):raise ValueError('Material exceeds frozen budget')
    rule=RULE.replace('只评匿名X/Y的候选原文能否覆盖这个问题的两条必需要点，不评学生、检索算法或实际投递。',
        '只评匿名X/Y两份最终教材投递计划中的原文能否覆盖问题的两条必需要点。每份材料在1200字符内；不能假定还有未列出的材料。本实验没有运行教学生成，不评学生、教学回答或学习效果。')
    rule+='\n另记录材料内OCR误读、残句和拼接风险；引用本组ID并说明是否妨碍某个必需要点。不要纠写原文，不能用自己的物理知识补足。'
    packets={};sources={}
    for reviewer,mapping in [('A',{'X':'baseline','Y':'selected'}),('B',{'X':'selected','Y':'baseline'})]:
        packet={'question':case['query'],'required_points':case['required_points']};facts={}
        for arm,name in mapping.items():
            facts[arm]={f'{arm}{n}':s for n,s in enumerate(plans[name]['sources'],1)}
            packet[arm]=[dict(id=i,source=s['source'],page=s['page'],type=s['type'],text=s['text']) for i,s in facts[arm].items()]
        packets[reviewer]={'mapping':mapping,'packet':packet};sources[reviewer]=facts
    protocol={'input_sha256':sha(input_path),'dataset_sha256':sha(dataset),'rule':rule,'packets':packets,
        'source_hashes':{n:sha(ROOT/n) for n in ['scripts/review_position_intro_material.py','scripts/review_rag_page_evidence.py','scripts/rag_bounded_model_call.py']},
        'scope':'Known development single case; two fresh same-family AI calls, no prior grades supplied',
        'max_model_calls':len(a.reviewers),'reviewers':a.reviewers,'request_timeout_s':a.request_timeout,
        'max_retries':0,'format_retries':0,'timeout_semantics':'SDK request timeout, not hard wall-clock deadline','defaults_changed':False}
    write(a.out/'protocol.json',protocol)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ.update(REVIEW_MAX_RETRIES='0',REVIEW_TIMEOUT=str(a.request_timeout))
    from app.config import get_settings
    from app.services.llm import get_llm,clear_cache
    get_settings.cache_clear();clear_cache()
    llm=get_llm('review')
    if llm.model!='deepseek-flash':raise ValueError('Model differs from authorization')
    llm=SingleAttemptStructuredClient(llm)
    protocol['effective_client_timeout_s']=llm.timeout
    write(a.out/'protocol.json',protocol)
    blocked=False;records=[]
    for reviewer in a.reviewers:
        record={'reviewer':reviewer,'status':'not_run','mapping':packets[reviewer]['mapping']}
        if not blocked:
            llm.reset_last_call()
            try:
                output=llm.chat_structured(rule,json.dumps(packets[reviewer]['packet'],ensure_ascii=False),Review).model_dump()
                record.update(status='valid',output=validate_review(output,sources[reviewer]))
            except Exception as error:
                record.update(status='failed',error=type(error).__name__+': '+str(error))
                blocked='402' in str(error) or 'Insufficient Balance' in str(error)
            record['call']=llm.last_call.as_dict() if llm.last_call else None
        else:record['reason']='Account failed; no further call'
        write(a.out/(reviewer+'.json'),record);records.append(record);print(reviewer,record['status'],flush=True)
    if sha(input_path)!=protocol['input_sha256'] or sha(dataset)!=protocol['dataset_sha256'] or any(sha(ROOT/n)!=h for n,h in protocol['source_hashes'].items()):raise RuntimeError('Frozen material/rule drift')
    conclusions=[]
    for record in records:
        if record['status']=='valid':
            conclusions.append({'reviewer':record['reviewer'],**{name:record['output'][arm]['sufficient'] for arm,name in record['mapping'].items()}})
    write(a.out/'summary.json',{'valid_reviews':len(conclusions),'conclusions':conclusions,
        'jointly_confirmed_single_case_improvement':len(conclusions)==2 and all(not c['baseline'] and c['selected'] for c in conclusions),
        'scope':'Development material evidence only; not default acceptance or teaching evaluation','defaults_changed':False})


if __name__=='__main__':main()
