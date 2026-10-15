"""Paired real selector calls on identical frozen texts; alternate arm order."""
import json,sys,argparse,hashlib
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from app.services.rag_selection import select,RULE,catalogue,selection_request
    from app.services.llm import get_llm
    sources=[ROOT/'artifacts/rag_selection_v3_development_20261014/results.json',ROOT/'artifacts/rag_selection_v3_validation_recovery_20261014/results.json']
    allrows=[r for p in sources for r in json.loads(p.read_text(encoding='utf-8'))]
    wanted=['sphere_energy','interference_intensity','index_speed_regression','disk_inertia','monatomic_heat','em_wave_vectors']
    rows=[next(r for r in allrows if r['id']==i) for i in wanted]
    files=['backend/app/services/rag_selection.py','backend/app/services/rag_materials.py']
    hashes={p:sha(ROOT/p) for p in files}
    write(a.out/'protocol.json',{'source_hashes':hashes,'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},'case_ids':wanted,'rule':RULE,'script_sha256':sha(Path(__file__)),
        'scope':'Known development cases, 6 paired calls per format; same original texts, alternating execution order; not causal speed proof or unseen validation',
        'activation':'No default switch in this experiment; retain all fallback outcomes for assessment'})
    results=[]
    for n,r in enumerate(rows):
        row={'id':r['id'],'question':r['question'],'required_points':r['required_points'],'arms':{},'status':'started'}
        write(a.out/(r['id']+'.json'),row)
        for arm in (['full_v1','compact_v2'] if n%2==0 else ['compact_v2','full_v1']):
            h=deepcopy(r['arms']['selected_evidence_v3']['hits']);pool=catalogue(h)
            llm=get_llm('tutor');plan=select(h,r['question'],llm=llm,input_format=arm)
            row['arms'][arm]={'plan':plan,'call':llm.last_call.as_dict() if llm.last_call else None}
            write(a.out/(r['id']+'.json'),row);print(r['id'],arm,plan['selection_status'],flush=True)
        row['status']='completed';write(a.out/(r['id']+'.json'),row);results.append(row)
    write(a.out/'results.json',results)
    write(a.out/'summary.json',{'cases':len(results),'source_drift':[f for f,s in hashes.items() if sha(ROOT/f)!=s],'default_changed':False})
if __name__=='__main__':main()
