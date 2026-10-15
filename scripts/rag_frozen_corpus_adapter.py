"""Transport-only frozen corpus adapter: real exported facts/vectors, no gold fabrication."""
import gzip,json,hashlib
from pathlib import Path

def frozen_stores(bundle,workdir):
    import numpy as np
    from sqlalchemy import create_engine
    from datetime import datetime
    from qdrant_client import QdrantClient,models as qm
    from app.db.tables import Base,DocumentBlock,DocumentImportRun
    import app.db.postgres as pg
    import app.rag.block_store as bs
    import app.rag.retriever as retriever
    bundle=Path(bundle);data=bundle/'deployment/data';manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    for p in data.iterdir():
        if not p.is_file():continue
        entry=manifest['files'][p.relative_to(bundle).as_posix()]
        assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
    engine=create_engine('sqlite:///'+str(Path(workdir)/'frozen_corpus.sqlite'));Base.metadata.create_all(engine)
    with engine.begin() as connection:
        for cls in [DocumentBlock,DocumentImportRun]:
            table=cls.__table__;batch=[]
            with gzip.open(data/(table.name+'.jsonl.gz'),'rt',encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    for col in table.columns:
                        if isinstance(row.get(col.name),str) and col.type.__class__.__name__=='DateTime':row[col.name]=datetime.fromisoformat(row[col.name])
                    batch.append(row)
                    if len(batch)==512:connection.execute(table.insert(),batch);batch=[]
                if batch:connection.execute(table.insert(),batch)
    cfg=json.loads((data/'qdrant_config.json').read_text(encoding='utf-8'));q=QdrantClient(':memory:');q.create_collection('textbook_chunks',vectors_config=qm.VectorParams(size=cfg['size'],distance=qm.Distance(cfg['distance'])))
    vectors=np.memmap(data/'vectors.f32',dtype='<f4',mode='r').reshape(-1,cfg['size']);batch=[];n=0
    with gzip.open(data/'qdrant_points.jsonl.gz','rt',encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            row=json.loads(line);batch.append(qm.PointStruct(id=row['id'],payload=row['payload'],vector=vectors[n-1].tolist()))
            if len(batch)==1024:q.upsert('textbook_chunks',batch);batch=[]
        if batch:q.upsert('textbook_chunks',batch)
    assert n==len(vectors)
    with engine.connect() as c:c.exec_driver_sql('PRAGMA query_only=ON')
    pg.get_engine=lambda:engine;bs.get_engine=lambda:engine;retriever.get_qdrant=lambda:q
    return engine,q,{'source':'verified deployment1.3 corpus; SQLite facts and Qdrant local exact search; same production retriever logic; not live PostgreSQL/HNSW validation','vectors':n,'manifest_sha256':hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()}
