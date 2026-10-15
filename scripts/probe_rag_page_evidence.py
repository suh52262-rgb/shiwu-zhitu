"""Prospective local retrieval and paired candidate-pool replay, no paid calls."""
import argparse,json,hashlib,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 dataset=ROOT/'evaluation/datasets/rag_page_evidence_probes_20261014.json';data=json.loads(dataset.read_text(encoding='utf-8-sig'))
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env')
 os.environ.update(RAG_RERANK='0',RAG_QUERY_EXPANSION='0',EMBEDDING_PROVIDER='bge',HF_HUB_OFFLINE='1',RAG_MATERIAL_POLICY='hit_first_total_v1',RAG_CONTEXT_POLICY='legacy')
 from sqlalchemy import event
 from app.db.postgres import get_engine
 engine=get_engine();event.listen(engine,'begin',lambda c:c.exec_driver_sql('SET TRANSACTION READ ONLY'))
 from app.rag.block_store import load_page_blocks
 from app.agent.tools import retrieve_knowledge
 from app.services.rag_selection import catalogue
 from app.rag.page_evidence_candidates import page_evidence_candidates
 files=['backend/app/rag/page_evidence_candidates.py','backend/app/rag/retriever.py','backend/app/services/rag_selection.py','backend/app/rag/block_store.py']
 hashes={f:sha(ROOT/f) for f in files}
 write(a.out/'protocol.json',{'dataset':data,'dataset_sha256':sha(dataset),'script_sha256':sha(Path(__file__)),'source_hashes':hashes,'scope':'Candidate pool only; local BGE, rerank/expansion off; read-only database; no teaching or actual delivery','fixed_process_overrides':{'rerank':False,'query_expansion':False,'embedding':'bge','material':'hit_first_total_v1','context':'legacy'},'default_changed':False})
 rows=[]
 for c in data['cases']:
  r={'id':c['id'],'question':c['query'],'required_points':c['required_points'],'status':'started'};write(a.out/(c['id']+'.json'),r)
  try:
   status={};start=time.perf_counter();hits=retrieve_knowledge(c['query'],top_k=5,status=status);r.update(retrieval_sec=time.perf_counter()-start,retrieval_status=status,hits=hits)
   base=catalogue(hits);facts={}
   keys={(h['source'],h['page']) for h in hits}|{(s['source'],s['page']) for s in base}
   for doc,page in sorted(keys):
    facts.update(load_page_blocks(doc,[page]))
   blocks=[{**b,'doc_id':doc,'source':doc} for (doc,page),bs in facts.items() for b in bs]
   start=time.perf_counter();added,omitted=page_evidence_candidates(base,hits,blocks);r['assembly_sec']=time.perf_counter()-start
   for s in base+added:
    b=next(b for b in blocks if b['id']==s['id'] and b['source']==s['source'] and b['page']==s['page'])
    assert b['text'].startswith(s['text']) and b['indexable'] is not False and not b['is_generated']
   assert sum(len(s['text']) for s in added)<=3600 and len({s['id'] for s in base+added})==len(base+added)
   r.update(status='completed',facts=blocks,added=added,omissions=omitted,arms={k:{'plan':{'sources':v,'content_characters':sum(len(s['text']) for s in v)}} for k,v in [('baseline_pool',base),('page_pool',base+added)]})
  except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
  write(a.out/(c['id']+'.json'),r);rows.append(r);print(c['id'],r['status'],flush=True)
 write(a.out/'results.json',rows);write(a.out/'summary.json',{'completed':sum(r['status']=='completed' for r in rows),'cases':len(rows),'source_drift':[f for f,h in hashes.items() if sha(ROOT/f)!=h],'paid_calls':0,'default_changed':False});engine.dispose()
if __name__=='__main__':main()
