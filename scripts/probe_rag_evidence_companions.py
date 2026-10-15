"""Prospective equation-reference companions; read-only facts/PDFs, no paid retrieval."""
import argparse,json,hashlib,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--snapshot',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 dataset=ROOT/'evaluation/datasets/rag_equation_companion_probes_20261014.json';data=json.loads(dataset.read_text(encoding='utf-8-sig'))
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env');os.environ.update(RAG_RERANK='0',RAG_QUERY_EXPANSION='0',EMBEDDING_PROVIDER='bge',HF_HUB_OFFLINE='1',RAG_MATERIAL_POLICY='hit_first_total_v1',RAG_CONTEXT_POLICY='legacy')
 from sqlalchemy import event
 from app.db.postgres import get_engine
 adapter=None
 if a.snapshot:
  from rag_frozen_corpus_adapter import frozen_stores
  engine,adapter,snapshot_provenance=frozen_stores(a.snapshot,a.out)
 else:
  engine=get_engine();event.listen(engine,'begin',lambda c:c.exec_driver_sql('SET TRANSACTION READ ONLY'));snapshot_provenance=None
 from app.rag.block_store import load_blocks
 from app.agent.tools import retrieve_knowledge
 from app.services.rag_selection import catalogue
 from app.rag.formula_positions import label_position_links,standalone_label
 from app.rag.formula_links import labels
 from app.rag.evidence_companions import evidence_companions,equation_references
 import fitz
 files=['backend/app/rag/evidence_companions.py','backend/app/rag/formula_positions.py','backend/app/rag/retriever.py','backend/app/services/rag_selection.py']
 hashes={f:sha(ROOT/f) for f in files};protocol={'dataset':data,'dataset_sha256':sha(dataset),'script_sha256':sha(Path(__file__)),'source_hashes':hashes,'retrieval_overrides':{'bge':True,'rerank':False,'expansion':False,'context':'legacy','material':'hit_first_total_v1'},'storage_snapshot':snapshot_provenance,'default_changed':False}
 write(a.out/'protocol.json',protocol);rows=[];corpus={};pdfs={};pdf_hashes={}
 for c in data['cases']:
  r={'id':c['id'],'question':c['query'],'required_points':c['required_points'],'status':'started'};write(a.out/(c['id']+'.json'),r)
  try:
   status={};t=time.perf_counter();hits=retrieve_knowledge(c['query'],top_k=5,status=status);base=catalogue(hits);r.update(hits=hits,retrieval_status=status,retrieval_sec=time.perf_counter()-t)
   for doc in {s['source'] for s in base}:
    if doc not in corpus:corpus[doc]=load_blocks(doc)
   facts=[b for doc in {s['source'] for s in base} for b in corpus[doc]]
   refs={(s['source'],n) for s in base for n in (equation_references(s['text']) if s['type']=='text' else labels(s))}
   pages={(b['doc_id'],b['page']) for b in facts if b['type']=='formula' and any((b['doc_id'],n) in refs for n in labels(b))}
   links,unknown=label_position_links([b for b in facts if (b['doc_id'],b['page']) in pages]);checked=[]
   for link in links:
    if (link['source'],link['reference']) not in refs:continue
    doc=link['source'];prefix=doc.split('-')[1];paths=list((ROOT/'data/textbooks').glob(f'教材-{prefix}-*.pdf'))
    if prefix=='1':paths=[p for p in paths if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
    if len(paths)!=1:link['source_check']='missing_pdf';checked.append(link);continue
    if doc not in pdfs:pdfs[doc]=fitz.open(paths[0]);pdf_hashes[paths[0].name]=sha(paths[0])
    page=pdfs[doc][link['page']-1];rect=fitz.Rect(link['label_bbox']);text=page.get_text('text',clip=rect+(-1,-1,1,1)).strip() if page.rect.contains(rect) else ''
    link.update(pdf_label_text=text,source_check='verified_label_text' if standalone_label(text)==link['reference'] else 'label_clip_mismatch');checked.append(link)
   t=time.perf_counter();added,omitted=evidence_companions(base,facts,checked);r['assembly_sec']=time.perf_counter()-t
   byid={b['id']:b for b in facts}
   for s in base+added:
    b=byid[s['id']];assert b['doc_id']==s['source'] and b['page']==s['page'] and b['text'].startswith(s['text']) and b['indexable'] and not b['is_generated']
   assert sum(len(s['text']) for s in added)<=1600 and len(added)<=8
   r.update(status='completed',printed_label_checks=checked,position_omissions=unknown,added=added,omissions=omitted,arms={k:{'plan':{'sources':v,'content_characters':sum(len(s['text']) for s in v)}} for k,v in [('baseline_pool',base),('companion_pool',base+added)]})
  except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e))
  write(a.out/(c['id']+'.json'),r);rows.append(r);print(c['id'],r['status'],len(r.get('added',[])),flush=True)
 for d in pdfs.values():d.close()
 write(a.out/'results.json',rows);write(a.out/'summary.json',{'cases':len(rows),'completed':sum(r['status']=='completed' for r in rows),'source_drift':[f for f,h in hashes.items() if sha(ROOT/f)!=h],'pdf_hashes':pdf_hashes,'corpus_hashes':{d:hashlib.sha256(json.dumps(bs,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for d,bs in corpus.items()},'paid_calls':0,'storage_snapshot':snapshot_provenance,'default_changed':False});engine.dispose()
 if adapter:adapter.close()
if __name__=='__main__':main()
