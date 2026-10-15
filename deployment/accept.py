"""Free deployment acceptance: integrity, services, counts, evidence and API.
No model calls, no writes to student tables. Online model smoke is manual.
"""
from pathlib import Path
import argparse,gzip,json,sys,os
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backend')]
from deployment_bundle import verify,DOCS
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
# Free infrastructure acceptance must not enter paid reranking/query expansion.
# This subprocess does not change the user's .env or production defaults.
os.environ['RAG_RERANK'] = '0'
os.environ['RAG_QUERY_EXPANSION'] = '0'
def main():
    from sqlalchemy import select,func
    from qdrant_client import QdrantClient
    from app.config import get_settings
    from app.db.postgres import get_engine
    from app.db.tables import Base,DocumentBlock
    # Register the separate evidence-review model before checking empty history.
    from app.services import error_evidence_reviews  # noqa: F401
    p=argparse.ArgumentParser();p.add_argument('--bundle',default=str(ROOT));a=p.parse_args()
    root=Path(a.bundle);result={'time':datetime.now().isoformat(),'paid_requests':0,'checks':{},'status':'failed'}
    try:
        verify(root);result['checks']['manifest']='passed'
        inv=json.loads((root/'DATA_INVENTORY.json').read_text(encoding='utf-8'));s=get_settings();engine=get_engine()
        with engine.connect() as c:
            counts={r[0]:r[1] for r in c.execute(select(DocumentBlock.doc_id,func.count()).group_by(DocumentBlock.doc_id))}
            assert counts=={d:v['blocks'] for d,v in inv['documents'].items()},counts
            result['checks']['postgres_blocks']=sum(counts.values())
            rows=c.execute(select(DocumentBlock.doc_id,func.count(func.distinct(DocumentBlock.page))).group_by(DocumentBlock.doc_id)).all()
            assert dict(rows)=={d:v['pages'] for d,v in inv['documents'].items()}
            # Fresh restore has no inherited students or attempts. After a user tries the app,
            # this check may fail intentionally; rerun before manual teaching acceptance.
            for name in ['students','attempts','learning_events','hint_records','student_profiles','analysis_traces','error_evidence_reviews']:
                assert c.execute(select(func.count()).select_from(Base.metadata.tables[name])).scalar()==0,name
            result['checks']['student_history_excluded']='passed'
            for doc_id in DOCS:
                b=c.execute(select(DocumentBlock.__table__).where(DocumentBlock.doc_id==doc_id).order_by(DocumentBlock.page,DocumentBlock.block_index).limit(1)).mappings().one()
                assert b['text'] or b['embed_text'] or b['figure_path']
        q=QdrantClient(url=s.qdrant_url,timeout=120,check_compatibility=False)
        assert q.count(s.qdrant_collection,exact=True).count==inv['vectors']
        result['checks']['qdrant_points']=inv['vectors']
        config=json.loads((root/'deployment/data/qdrant_config.json').read_text(encoding='utf-8'));sample=[];expected={}
        with gzip.open(root/'deployment/data/qdrant_points.jsonl.gz','rt',encoding='utf-8') as f,(root/'deployment/data/vectors.f32').open('rb') as v:
            for i,line in enumerate(f):
                row=json.loads(line);raw=v.read(config['size']*4)
                if i%4096==0:
                    sample.append(row['id']);expected[str(row['id'])]=(row['payload'],np.frombuffer(raw,dtype='<f4'))
        pts=q.retrieve(s.qdrant_collection,ids=sample,with_payload=True,with_vectors=True)
        assert len(pts)==len(sample)
        for pt in pts:
            payload,vec=expected[str(pt.id)];assert pt.payload==payload
            assert np.allclose(vec,np.asarray(pt.vector),rtol=1e-6,atol=1e-7)
        result['checks']['point_payload_vector_samples']=len(sample);q.close()
        from app.rag.embedding import get_embedding_client
        from app.rag.retriever import retrieve
        embed=get_embedding_client().embed(['动量守恒']);assert len(embed[0])==config['size']
        hits=retrieve('动量守恒',top_k=3)
        assert hits and any(h.get('context_blocks') for h in hits),hits
        for hit in hits:
            assert hit['source'] in DOCS and hit.get('page') is not None
        result['checks']['rag_search']={'hits':len(hits),'sources':[h['source'] for h in hits],'context_expansion':True}
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client:
            assert client.get('/health').status_code==200
            problems=client.get('/problems');assert problems.status_code==200
        result['checks']['api_health_problem_list']='passed'
        result['status']='passed';engine.dispose()
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        target=root/'deployment/acceptance-result.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
