"""Read-only stage audit of saved RAG hits; no retrieval, model calls or gold-guided selection."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
def digest(data):return hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env')
 from sqlalchemy import event
 from app.db.postgres import get_engine
 engine=get_engine();event.listen(engine,'begin',lambda c:c.exec_driver_sql('SET TRANSACTION READ ONLY'))
 from app.rag.block_store import load_blocks
 from app.rag.formula_links import labels
 from app.services.rag_selection import catalogue
 from rag_candidate_trace import trace_block
 from app.rag.reference_candidates import document_reference_candidates
 files=[ROOT/'artifacts'/n/'results.json' for n in ['rag_selection_v3_development_20261014','rag_selection_v3_validation_recovery_20261014']]
 originals=[r for p in files for r in json.loads(p.read_text(encoding='utf-8'))]
 ids=['wave_positive','spherical_cap','monatomic_heat','rod_emf','dielectric_supply','em_wave_vectors','oscillator_energy']
 rows=[next(r for r in originals if r['id']==i) for i in ids]
 docs=sorted({h['source'] for r in rows for h in r['arms']['selected_evidence_v3']['hits']});facts={d:load_blocks(d) for d in docs}
 write(a.out/'protocol.json',{'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256((ROOT/'scripts/rag_candidate_trace.py').read_bytes()).hexdigest(),'prototype_sha256':hashlib.sha256((ROOT/'backend/app/rag/reference_candidates.py').read_bytes()).hexdigest(),'corpus_hashes':{d:digest(bs) for d,bs in facts.items()},'scope':'Saved five-hit outputs and same-document facts; no new ranking or model requests; read-only transactions; no semantic sufficiency inference'})
 results=[]
 for r in rows:
  hits=r['arms']['selected_evidence_v3']['hits'];pool=catalogue(hits);poolids={s['id'] for s in pool};detail=[];nearby={}
  for h in hits:
   bs=facts[h['source']];pages={h['page']+i for i in range(-2,3)};around=[b for b in bs if b['page'] in pages]
   nearby.update({b['id']:{**b,'source':h['source']} for b in around})
   issues=[]
   for omission in h.get('context_link_omissions',[]):
    number=omission['reference'];matches=[b for b in bs if b['type']=='formula' and number in labels(b)]
    eligible=[b for b in matches if b['page'] in {h['page'],h['page']-1}]
    if len(eligible)>1:reason='ambiguous_within_page_scope'
    elif eligible and (eligible[0]['indexable'] is False or eligible[0]['is_generated']):reason='filtered_in_page_scope'
    elif eligible:reason='present_in_page_scope_but_omitted'
    elif matches:reason='found_outside_same_previous_page'
    else:reason='no_labelled_formula_in_same_document'
    issues.append({**omission,'stage_reason':reason,'matches':[{'id':b['id'],'page':b['page'],'block_index':b['block_index'],'indexable':b['indexable'],'is_generated':b['is_generated'],'text':b['text']} for b in matches]})
   detail.append({'hit_id':h.get('block_id'),'source':h['source'],'page':h['page'],'block_index':h.get('block_index'),'text':h['text'],'reference_omissions':issues})
  proposed, rejected = document_reference_candidates(pool,[b for bs in facts.values() for b in bs])
  record={'proposed_document_references':proposed,'document_reference_omissions':rejected,'id':r['id'],'question':r['question'],'required_points':r['required_points'],'hits':detail,'candidate_ids':sorted(poolids),'nearby_blocks':[dict(b,in_candidate=b['id'] in poolids,stage_trace=trace_block(b,hits,pool)) for b in sorted(nearby.values(),key=lambda b:(b['source'],b['page'],b['block_index']))]}
  write(a.out/(r['id']+'.json'),record);results.append({k:record[k] for k in ['id','hits','proposed_document_references','document_reference_omissions']});print(r['id'],sum(len(h['reference_omissions']) for h in detail),len(nearby),flush=True)
 write(a.out/'summary.json',{'cases':results,'corpus_block_counts':{d:len(bs) for d,bs in facts.items()},'default_changed':False,'paid_calls':0});engine.dispose()
if __name__=='__main__':main()
