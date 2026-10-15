"""Frozen new-query retrieval and source-bound formula pools; no paid calls."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    os.environ.update(RAG_RERANK='0',RAG_QUERY_EXPANSION='0',EMBEDDING_PROVIDER='bge',
        HF_HUB_OFFLINE='1',RAG_CONTEXT_POLICY='legacy',RAG_MATERIAL_POLICY='hit_first_total_v1')
    model=a.snapshot/'models/bge-small-zh-v1.5'
    manifest=json.loads((a.snapshot/'manifest.json').read_text(encoding='utf-8'))
    model_hashes={}
    for path in model.rglob('*'):
        if path.is_file():
            key=path.relative_to(a.snapshot).as_posix();digest=sha(path)
            if manifest['files'][key]['sha256']!=digest:raise ValueError('Model identity mismatch')
            model_hashes[key]=digest
    if not model_hashes or not (model/'modules.json').is_file():raise ValueError('Missing snapshot model')
    os.environ['EMBEDDING_MODEL']=str(model.resolve())
    dataset=ROOT/'evaluation/datasets/rag_region_queries_20261014.json'
    data=json.loads(dataset.read_text(encoding='utf-8'));cases=data['cases'];ids=[c['id'] for c in cases]
    if len(ids)!=4 or len(set(ids))!=4 or any(not c['query'].strip() for c in cases):raise ValueError('Invalid frozen cases')
    audit=json.loads((a.audit/'protocol.json').read_text(encoding='utf-8'))
    checks=json.loads((a.audit/'checks.json').read_text(encoding='utf-8'))
    positions=[dict(c,source_check='verified_label_text') for c in checks if c['result']['status']=='verified']
    for info in audit['pdfs'].values():
        if sha(Path(info['path']))!=info['sha256']:raise ValueError('Original PDF changed')
    sources=['scripts/probe_rag_region_queries.py','scripts/rag_frozen_corpus_adapter.py',
        'backend/app/rag/retriever.py','backend/app/rag/evidence_companions.py',
        'backend/app/services/rag_selection.py','backend/app/services/rag_materials.py']
    protocol={'dataset_sha256':sha(dataset),'dataset':data,'expected_case_ids':ids,
        'sources':{s:sha(ROOT/s) for s in sources},'model_files':model_hashes,
        'audit_checks_sha256':sha(a.audit/'checks.json'),'required_points_to_assembler':False,
        'paid_calls':0,'defaults_changed':False}
    write(a.out/'protocol.json',protocol)
    from rag_frozen_corpus_adapter import frozen_stores
    engine,adapter,storage=frozen_stores(a.snapshot,a.out)
    from app.agent.tools import retrieve_knowledge
    from app.rag.block_store import load_blocks
    from app.rag.evidence_companions import evidence_companions
    from app.services.rag_materials import fragments
    from app.services.rag_selection import catalogue
    protocol['storage']=storage
    rows=[];corpus={}
    try:
        for case in cases:
            row={'id':case['id'],'query':case['query'],'required_points':case['required_points'],'status':'started'}
            try:
                started=time.perf_counter();status={}
                hits=retrieve_knowledge(case['query'],top_k=5,status=status)
                baseline=fragments(hits,selected_policy='hit_first_total_v1');pool=catalogue(hits)
                facts=[]
                for doc in sorted({b['source'] for b in pool}):
                    if doc not in corpus:corpus[doc]=load_blocks(doc)
                    facts.extend(corpus[doc])
                byid={b['id']:b for b in facts}
                for source in pool:
                    b=byid[source['id']]
                    if (b['doc_id'],b['page'],b['type'])!=(source['source'],source['page'],source['type']) or not b['text'].startswith(source['text']) or not b['indexable'] or b['is_generated']:
                        raise ValueError('Candidate source/text mismatch')
                used=sum(len(b['text']) for b in pool)
                if used>12000 or len(pool)>48:raise ValueError('Baseline pool exceeds budget')
                added,omitted=evidence_companions(pool,facts,positions,budget=12000-used,limit=48-len(pool))
                row.update(status='completed',hits=hits,retrieval_status=status,baseline_plan=baseline,
                    candidate_pool=pool+added,formula_added=added,omissions=omitted,
                    candidate_characters=used+sum(len(b['text']) for b in added),seconds=time.perf_counter()-started,
                    actual_candidate_selection_evaluated=False)
            except Exception as error:row.update(status='failed',error=type(error).__name__+': '+str(error))
            write(a.out/(case['id']+'.json'),row);rows.append(row)
            print(case['id'],row['status'],'formula additions',len(row.get('formula_added',[])),flush=True)
        protocol['database_sha256']=sha(a.out/'frozen_corpus.sqlite')
        if sha(dataset)!=protocol['dataset_sha256'] or any(sha(ROOT/n)!=h for n,h in protocol['sources'].items()):raise RuntimeError('Frozen source/dataset drift')
        write(a.out/'protocol.json',protocol);write(a.out/'results.json',rows)
        write(a.out/'summary.json',{'records':len(rows),'completed':sum(r['status']=='completed' for r in rows),
            'paid_calls':0,'defaults_changed':False,'semantic_sufficiency_evaluated':False})
    finally:
        engine.dispose();adapter.close()


if __name__=='__main__':main()
