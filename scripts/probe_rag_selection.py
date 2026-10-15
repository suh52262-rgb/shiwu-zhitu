"""Saved-query development comparison: selector receives query only, never gold."""
import argparse,json,sys,hashlib
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--scope',choices=['development','validation'],default='development');a=ap.parse_args();a.out.mkdir(exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from app.services.rag_selection import select,catalogue,RULE
    from app.services.rag_materials import fragments
    from app.services.llm import get_llm
    data=json.loads((a.input/'results.json').read_text(encoding='utf-8'));files=['backend/app/services/rag_selection.py','backend/app/services/rag_materials.py','backend/app/rag/formula_links.py','backend/app/agent/tools.py']
    hashes={f:sha(ROOT/f) for f in files}
    write(a.out/'protocol.json',{'input_sha256':sha(a.input/'results.json'),'source_hashes':hashes,'script_sha256':sha(Path(__file__)),'rule':RULE,'budget':1200,
       'scope':a.scope+' on frozen supplied queries; selection uses no labels, required points, ratings or ideal solutions',
       'activation':'Requires separately frozen new queries and no confirmed semantic regressions; this development run alone cannot enable default'})
    rows=[]
    for original in data:
        row={'id':original['id'],'question':original['question'],'required_points':original['required_points'],'status':'started'};write(a.out/(row['id']+'.json'),row)
        h=deepcopy(original['arms']['formula_refs_v2']['hits']);llm=get_llm('tutor');plan=select(h,original['question'],llm=llm)
        pool=catalogue(h);facts={s['id']:s for s in pool}
        assert all(s['id'] in facts and s['text'] in facts[s['id']]['text'] for s in plan['sources'])
        assert plan['content_characters']<=1200
        h[0]['_selected_material']=plan
        delivered=fragments(h,selected_policy='selected_evidence_v3')
        assert delivered['sources']==plan['sources']
        row.update(status='completed',selection_status=plan['selection_status'],selection_candidates=pool,
                   arms={'legacy':original['arms']['legacy'],'selected_evidence_v3':{'hits':h,'plan':delivered}},
                   call=llm.last_call.as_dict() if llm.last_call else None)
        write(a.out/(row['id']+'.json'),row);rows.append(row);print(row['id'],plan['selection_status'],flush=True)
    write(a.out/'results.json',rows)
    write(a.out/'summary.json',{'records':len(rows),'ready':sum(r['selection_status']=='ready' for r in rows),'failed':sum(r['selection_status']=='failed' for r in rows),'source_drift':[f for f,v in hashes.items() if sha(ROOT/f)!=v],'default_enabled':False})
if __name__=='__main__':main()
