"""Two fresh same-family AI reviews of frozen actual delivery, not teacher review."""
import argparse,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from validate_rag_materials import Assessment,RUBRIC

class Item(BaseModel):
    id:str
    X:Assessment
    Y:Assessment
class Batch(BaseModel):
    items:list[Item]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--candidate',default='formula_refs_v1',choices=['formula_refs_v1','formula_refs_v2','selected_evidence_v3','candidate_pool_audit','compact_v2']);ap.add_argument('--baseline',default='legacy');ap.add_argument('--scope',choices=['delivery','pool_diagnostic'],default='delivery');a=ap.parse_args()
    rows=json.loads((a.comparison/'results.json').read_text(encoding='utf-8'));a.out.mkdir(exist_ok=False)
    packets=[];mapping={}
    for n,r in enumerate(rows):
        arms=[a.baseline,a.candidate] if n%2 else [a.candidate,a.baseline]
        mapping[r['id']]=dict(zip(('X','Y'),arms))
        packets.append({'id':r['id'],'question':r['question'],'required_points':r['required_points'],
                        **{k:r['arms'][v]['plan']['sources'] for k,v in mapping[r['id']].items()}})
    rule=RUBRIC+'\n材料为固定命中块的实际投递片段。只判具体问题及两条要点；新增编号公式不自动算充分。明确指出新增材料是否无关，或挤掉必要正文造成退化。引用须逐字存在于本题本组text中。每个Assessment.point_status只能covered/partial/missing，数量=要点数；sufficient必须等于全部covered。'
    if a.scope=='pool_diagnostic':
        rule=RUBRIC+'\n本次只是诊断：一组是已投递材料，一组是未裁预算的完整候选目录。只判材料是否含所问两条要点，不把候选充分说成已投递充分，也不考虑1200预算。每个point_status仅covered/partial/missing，数量为2；sufficient等于两点均covered。引文必须在本题本组text中逐字存在。'
    write(a.out/'protocol.json',{'comparison_sha256':sha(a.comparison/'results.json'),'script_sha256':sha(Path(__file__)),'rule':rule,'independence':'fresh same-family calls; no shared ratings or implementation; not human evaluation','mapping':mapping})
    def review(name):
        from app.services.llm.factory import get_llm
        llm=get_llm('review');record={'stream':name,'status':'started'}
        try:
            output=llm.chat_structured(rule,json.dumps(packets,ensure_ascii=False),Batch).model_dump();record['output']=output
            assert len(output['items'])==len(rows) and {i['id'] for i in output['items']}==set(mapping)
            byid={p['id']:p for p in packets}
            for i in output['items']:
                for arm in ('X','Y'):
                    d=i[arm];assert len(d['point_status'])==2 and all(v in {'covered','partial','missing'} for v in d['point_status'])
                    assert d['sufficient']==all(v=='covered' for v in d['point_status'])
                    assert all(q and any(q in s['text'] for s in byid[i['id']][arm]) for q in d['supporting_quotes'])
                    assert not d['sufficient'] or d['supporting_quotes']
            record['status']='valid'
        except Exception as e:record.update(status='failed',error=type(e).__name__+': '+str(e))
        record['call']=llm.last_call.as_dict() if llm.last_call else None;write(a.out/(name+'.json'),record)
        print(name,record['status'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(review,n) for n in ('A','B')]
        for f in fs:f.result()
if __name__=='__main__':main()
