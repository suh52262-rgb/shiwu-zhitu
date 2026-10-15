"""Read-only semantic recheck of previously delivered fragments; no retrieval/tuning."""
import argparse,hashlib,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from pydantic import BaseModel,Field
from app.services.llm.factory import get_llm


class Citation(BaseModel):
    material_id:str
    quote:str
class Point(BaseModel):
    point_index:int
    status:str=Field(description='covered / partial / missing / unclear')
    explanation:str
    citations:list[Citation]
class Check(BaseModel):
    points:list[Point]
    sufficient:bool
    explanation:str


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
RULE=('只评所提供教材片段能否覆盖预定全部必需要点。不得引用常识、参考解、系统提示或其他页补全教材。'
      '明确区分结论、条件与机制；提到概念不等于说明机制。每个covered/partial必须引用实际片段ID及逐字连续子串。'
      'missing/unclear不得编引文。point_index按0起，必须每个要点恰好一条。sufficient当且仅当全部covered。'
      '只对这一轮材料下结论，未覆盖不等于全库不存在。保留OCR歧义，不自行复原公式。')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    records=[(x,json.loads(x.read_text(encoding='utf-8'))) for x in sorted(a.source.glob('*.json')) if x.name not in ('summary.json','protocol.json')]
    assert len(records)==6
    protocol={'rule':RULE,'source_files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x,d in records},
              'scope':'fresh same-family AI rechecks, no original scores provided, no retrieval, no teacher claim'}
    write(a.out/'protocol.json',protocol)
    def run(stream):
        for path,d in records:
            materials=[{'material_id':f'M{i}','source':x['source'],'page':x['page'],'text':x['text'],'truncated':x.get('truncated',False)} for i,x in enumerate(d['prompt_materials'])]
            packet={'question':d['case']['question'],'required_points':d['case']['points'],'materials':materials}
            row={'case_id':d['case']['id'],'stream':stream,'status':'failed','packet':packet}
            try:
                llm=get_llm('review');r=llm.chat_structured(RULE,json.dumps(packet,ensure_ascii=False),Check);v=r.model_dump();row['raw']=v
                by_id={m['material_id']:m['text'] for m in materials}
                assert len(v['points'])==len(packet['required_points']) and {q['point_index'] for q in v['points']}==set(range(len(packet['required_points'])))
                for point in v['points']:
                    assert point['status'] in ('covered','partial','missing','unclear')
                    if point['status'] in ('covered','partial'):assert point['citations']
                    for c in point['citations']:assert c['quote'] and c['quote'] in by_id[c['material_id']]
                assert v['sufficient']==all(q['status']=='covered' for q in v['points'])
                row.update(status='verified',call=llm.last_call.as_dict() if llm.last_call else None)
            except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:300]
            write(a.out/(d['case']['id']+'_'+stream+'.json'),row)
            print(stream,d['case']['id'],row['status'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,('C','D')))
    assert all(hashlib.sha256(Path(x).read_bytes()).hexdigest()==h for x,h in protocol['source_files'].items())
if __name__=='__main__':main()
