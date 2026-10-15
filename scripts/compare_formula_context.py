"""Replay production context and delivery on immutable page facts; no models."""
import argparse,json,hashlib,sys
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.rag import retriever,block_store
from app.services.rag_materials import fragments

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    data=json.loads(a.frozen.read_text(encoding='utf-8'))
    assert all(sha(ROOT/f)==s for f,s in data['source_hashes'].items())
    a.out.mkdir(exist_ok=False);results=[]
    def load(doc,pages):return {(doc,page):deepcopy(data['page_blocks'].get(doc+':'+str(page),[])) for page in pages}
    for case in data['cases']:
        row={'id':case['id'],'question':case['question'],'required_points':case['required_points'],'arms':{}}
        for policy in ('legacy','formula_refs_v1'):
            hit=case['hit'];hits=[{'source':case['source'],'page':case['page'],'text':hit['text'],'block_id':hit['id']}]
            with patch.object(block_store,'load_page_blocks',load):
                retriever._expand_context(hits,context_policy=policy)
            plan=fragments(hits,selected_policy='legacy');facts={b['id']:b for blocks in data['page_blocks'].values() for b in blocks}
            for s in plan['sources']:
                assert s['id'] in facts and s['text'] in facts[s['id']]['text']
                assert s['page']==facts[s['id']]['page']
                assert facts[s['id']]['indexable'] is not False and not facts[s['id']]['is_generated']
            row['arms'][policy]={'hits':hits,'plan':plan,'formula_delivered':any(s['id']==case['required_formula_id'] for s in plan['sources'])}
        old,new=(row['arms'][k] for k in ('legacy','formula_refs_v1'))
        ids={s['id'] for s in new['plan']['sources']}
        row['displaced_legacy_fragments']=[s for s in old['plan']['sources'] if s['id'] not in ids]
        results.append(row)
    summary={'cases':len(results),'legacy_formula_delivered':sum(r['arms']['legacy']['formula_delivered'] for r in results),'candidate_formula_delivered':sum(r['arms']['formula_refs_v1']['formula_delivered'] for r in results),'cases_with_displaced_fragments':sum(bool(r['displaced_legacy_fragments']) for r in results),'frozen_sha256':sha(a.frozen),'script_sha256':sha(Path(__file__)),'source_drift':[f for f,s in data['source_hashes'].items() if sha(ROOT/f)!=s],'default_changed':False,'semantic_sufficiency_verified':False,'scope':'Fixed-hit expansion/delivery, not vector recall or overall RAG accuracy'}
    (a.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
