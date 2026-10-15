"""Two independent fresh AI review streams, with immutable initial records.

Reviewers receive anonymous packets and frozen rules, never implementation,
restricted category mapping or each other's scores. Same-family AI reviews do
not constitute teacher approval. No failed or malformed review is zero-filled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from pydantic import BaseModel, Field
from run_t4_batch import digest, evidence_digest

RUBRIC='''你是独立AI评审，只根据本次投递的匿名包逐条评输出。不得推测实现版本，不能跨包搬用证据。
完整执行各包scoring_rules：前三项越高越好；answer_leakage越高越差、无泄题为1，只管teaching.hint；issue管认知归因，但不管泄题。评报告/提示，不是给材料正确程度评分。
ratable必须等于workflow_status==succeeded；不可评项四项评分为null，仍保留所有记录。review=null时explicit_error_detected=false。
四项严重指标：严重误指控、无依据认知断言、不可定位引用、严重物理矛盾；无证据允许false，不能靠关键词判违规。
认知归因判据：可以指认纸上错误的关系/运算/结论，禁止断言理解状态、信念、动机或未展示的推导过程。issue没有字段豁免。即使结果错、有其他纸面依据，也不能把认识状态当作观察事实。
求助、提示强度、缺少步骤、画面停留不能自行证明认知薄弱或卡住。书面排列、提交先后，不等于真实推导先后或因果依赖；只有结论值不能反推出学生解题路径。
核对时重建独立标记的系统推导不当作学生步骤；合法的大小加方向表达不能当错误；信息不足不能当确定错误。
引用须在所称来源中逐字存在；普通解释的术语引号不自动算作材料归属引用。引用正确不等于判断成立。
每条evidence说明具体字段及原文支持。四项严重flag为true必须给本包中的逐字quoted_evidence，不得引另一包。
material_has_definite_error评学生原文是否有明确物理/运算错误；explicit_error_detected评复盘是否识别该错。评分锚点沿用包内冻结说明，不发明通过门槛。'''


class Rating(BaseModel):
    blind_item_id: str
    ratable: bool
    correctness_1to5: int | None
    targeting_1to5: int | None
    actionability_1to5: int | None
    answer_leakage_1to5: int | None
    material_has_definite_error: bool
    explicit_error_detected: bool
    severe_false_accusation: bool
    unsupported_cognitive_claim: bool
    invalid_displayed_reference: bool
    serious_physics_contradiction: bool
    evidence: str
    quoted_evidence: list[str]=Field(default_factory=list)


class Group(BaseModel):
    items:list[Rating]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def strings(value):
    if isinstance(value,str):return [value]
    if isinstance(value,list):return [s for v in value for s in strings(v)]
    if isinstance(value,dict):return [s for v in value.values() for s in strings(v)]
    return []



def quotes_valid(packet, quotes):
    corpus = strings({key:packet.get(key) for key in
                      ('teaching','review','student_material','statement','reference_solution')})
    return all(bool(q.strip()) and any(q in source for source in corpus) for q in quotes)


def reviewer(name,packets,out,bindings):
    from app.config import get_settings
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    from app.services.llm.circuit import AccountCircuit, attach_guard
    settings=get_settings()
    guard=AccountCircuit('deepseek',settings.deepseek_base_url,settings.deepseek_api_key)
    ratings=[]
    for index in range(0,len(packets),6):
        group=packets[index:index+6]
        record={'reviewer_id':name,'item_ids':[p['blind_item_id'] for p in group],'status':'started'}
        target=out/f'{name}_group_{index//6+1:02d}.json'
        if guard.status()['blocked']:
            record.update(status='blocked',error='Account requires explicit local recovery; no reviewer request executed')
            write(target,record)
            continue
        judge=attach_guard(OpenAICompatibleProvider(api_key=settings.deepseek_api_key,base_url=settings.deepseek_base_url,
             model='deepseek-flash',temperature=0.2,timeout=120,max_retries=1,name='deepseek'),
             'deepseek',settings.deepseek_base_url,settings.deepseek_api_key)
        try:
            response=judge.chat_structured(RUBRIC,json.dumps(group,ensure_ascii=False),Group)
            items=[r.model_dump() for r in response.items]
            record['raw_items']=items
            by_id={p['blind_item_id']:p for p in group}
            if len(items)!=len(group) or {i['blind_item_id'] for i in items}!=set(by_id):
                raise ValueError('Missing, duplicate or cross-packet item identity')
            for item in items:
                packet=by_id[item['blind_item_id']]
                if item['ratable']!=(packet['workflow_status']=='succeeded'):
                    raise ValueError('Ratable decision conflicts with frozen rule')
                if packet.get('review') is None and item['explicit_error_detected']:
                    raise ValueError('Null review cannot identify an error')
                for key in ('correctness_1to5','targeting_1to5','actionability_1to5','answer_leakage_1to5'):
                    if (item['ratable'] and (type(item[key]) is not int or not 1<=item[key]<=5)
                            or not item['ratable'] and item[key] is not None):
                        raise ValueError('Invalid score scale')
                alleged=any(item[k] for k in ('severe_false_accusation','unsupported_cognitive_claim',
                               'invalid_displayed_reference','serious_physics_contradiction'))
                if alleged and not item['quoted_evidence']:
                    raise ValueError('Severe allegation lacks a literal quotation')
                if not quotes_valid(packet, item['quoted_evidence']):
                    raise ValueError('Reviewer quotation not present in this packet')
            ratings.extend(items)
            record['status']='completed'
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        finally:
            record['call']=judge.last_call.as_dict() if judge.last_call else None
            write(target,record)
        print(name,index//6+1,record['status'],flush=True)
    if len(ratings)!=72:
        write(out/f'{name}_incomplete.json',{'recorded':len(ratings),'expected':72,'status':'not_a_complete_review'})
        return
    # Set-level gates are computed only after blind initial ratings are locked.
    write(out/f'{name}_initial.json', {**bindings,'reviewer_id':name,'provenance':'independent_ai_review',
         'initial_ratings_locked':True,'blind_item_ids':sorted(i['blind_item_id'] for i in ratings),
         'scope':'fresh same-family AI calls; no teacher claim','items':ratings})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wait-for-batch',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest_hash=digest(args.manifest)
    write(args.out/'review_protocol.json',{'rubric':RUBRIC,'rubric_sha256':hashlib.sha256(RUBRIC.encode()).hexdigest(),
          'manifest_sha256':manifest_hash,'implementation_sha256':digest(__file__),
          'independence':'Separate streams, fresh requests, no shared ratings or implementation context',
          'scope':'Same-family AI provisional reviews, not human assessments', 'gating_criteria':'unchanged frozen development gate'})
    start=time.monotonic()
    while not (args.batch/'summary.json').exists():
        if not args.wait_for_batch or time.monotonic()-start>7200:
            raise RuntimeError('Complete batch required before judging')
        time.sleep(2)
    packets=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((args.batch/'review_packets').glob('Q*.json'))]
    if json.loads((args.batch/'summary.json').read_text(encoding='utf-8')).get('blocked',0):
        raise RuntimeError('Batch contains unexecuted blocked cases; no paid reviews started')
    if len(packets)!=72:raise ValueError('72 anonymous packets required')
    bindings={'manifest_sha256':manifest_hash,'summary_sha256':digest(args.batch/'summary.json'),
              'evidence_sha256':evidence_digest(args.batch)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(reviewer,name,packets,args.out,bindings) for name in ('ai_stream_A','ai_stream_B')]
        for future in futures:future.result()


if __name__=='__main__':main()
