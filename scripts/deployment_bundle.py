"""Portable, student-free deployment export/restore. Never exports arbitrary tables.

Vector records and float32 matrix share row order; manifest hashes protect both.
Restore requires empty target tables and an absent collection, never deletes data.
The package uses current workspace bytes, not git HEAD alone.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.metadata as meta, json, os, platform
from pathlib import Path
import shutil, subprocess, sys, zipfile
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
DOCS=[f'jiaocai-{s}' for s in ['1-jieda','1-shangce','2-xiache','3-lixue','4-rexue','5-dianci','6-guangxue','7-liangzi']]
TABLES=['document_blocks','document_import_runs']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def json_default(obj):
    if isinstance(obj,datetime):return obj.isoformat()
    raise TypeError(type(obj).__name__)
def app_imports():
    sys.path.insert(0,str(ROOT/'backend'))
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
def export_data(out):
    import numpy as np
    from sqlalchemy import select
    app_imports()
    from app.db.tables import DocumentBlock,DocumentImportRun
    from app.db.postgres import get_engine
    from app.db.qdrant import get_qdrant
    from app.config import get_settings
    from qdrant_client import models as qm
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    stats={d:{'blocks':0,'pages':set(),'vectors':0} for d in DOCS}
    engine=get_engine()
    # One consistent PostgreSQL read transaction. No students/attempts/traces involved.
    with engine.connect().execution_options(isolation_level='REPEATABLE READ') as c:
        with c.begin():
            for cls in [DocumentBlock,DocumentImportRun]:
                tab=cls.__table__;n=0
                with gzip.open(out/(tab.name+'.jsonl.gz'),'wt',encoding='utf-8',newline='\n') as f:
                    rows=c.execution_options(stream_results=True).execute(select(tab).where(tab.c.doc_id.in_(DOCS)).order_by(tab.c.id)).mappings()
                    for row in rows:
                        value=dict(row)
                        if cls==DocumentImportRun:
                            value['source_name']=str(value['source_name'] or '').replace('\\','/').split('/')[-1]
                            value['error']='' # diagnostics may contain local environment details
                        else:
                            # Normalize only asset locations; actual texts and identifiers remain exact.
                            value['figure_path']=portable_figure(value.get('figure_path',''))
                            d=stats[value['doc_id']];d['blocks']+=1;d['pages'].add(value['page'])
                        f.write(json.dumps(value,ensure_ascii=False,default=json_default)+'\n');n+=1
                print(tab.name,n,flush=True)
    q=get_qdrant();settings=get_settings();info=q.get_collection(settings.qdrant_collection)
    cfg=info.config.params.vectors
    if isinstance(cfg,dict):raise ValueError('Named vectors unsupported by this bundle format')
    write(out/'qdrant_config.json',{'collection':settings.qdrant_collection,'size':cfg.size,'distance':cfg.distance.value,'source_config':info.model_dump(mode='json'),'embedding_provider':settings.embedding_provider,'embedding_model':settings.embedding_model})
    filt=qm.Filter(must=[qm.FieldCondition(key='source',match=qm.MatchAny(any=DOCS))])
    offset=None;n=0
    with gzip.open(out/'qdrant_points.jsonl.gz','wt',encoding='utf-8',newline='\n') as f,(out/'vectors.f32').open('wb') as v:
        while True:
            points,offset=q.scroll(settings.qdrant_collection,scroll_filter=filt,limit=512,offset=offset,with_vectors=True,with_payload=True)
            for pt in points:
                vec=np.asarray(pt.vector,dtype='<f4')
                if vec.shape!=(cfg.size,):raise ValueError('Vector dimension mismatch')
                payload=dict(pt.payload or {});payload['figure_path']=portable_figure(payload.get('figure_path',''))
                f.write(json.dumps({'id':pt.id,'payload':payload},ensure_ascii=False)+'\n');v.write(vec.tobytes());n+=1
                stats[payload['source']]['vectors']+=1
            if n%10240==0:print('vectors',n,flush=True)
            if offset is None:break
    q.close();engine.dispose()
    for d in stats.values():d['pages']=len(d['pages'])
    if any(d['blocks']==0 or d['vectors']==0 for d in stats.values()):raise ValueError('Missing source document')
    write(out/'inventory.json',{'documents':stats,'document_count':len(stats),'blocks':sum(d['blocks'] for d in stats.values()),'pages':sum(d['pages'] for d in stats.values()),'vectors':n,'tables':TABLES,'excluded':['students','attempts','learning_events','hint_records','student_profiles','analysis_traces','video_jobs','video_runs'],'vector_encoding':'little-endian float32, ordered identically to qdrant_points.jsonl.gz'})
    print('DATA_EXPORT_OK',n,flush=True)
def portable_figure(value):
    value=str(value or '').replace('\\','/')
    if not value:return ''
    marker='data/documents/'
    if value.startswith('figures/'):
        if '..' in Path(value).parts:raise ValueError('Unsafe asset path')
        return value
    if marker not in value:raise ValueError('Unexpected figure path outside data/documents')
    value=marker+value.split(marker,1)[1]
    if '..' in Path(value).parts:raise ValueError('Unsafe asset path')
    return value
def verify(bundle):
    bundle=Path(bundle).resolve();m=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    for rel,entry in m['files'].items():
        p=(bundle/rel).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('Package integrity mismatch: '+rel)
    print('PACKAGE_VERIFY_OK',len(m['files']),flush=True);return m
def upsert_batch(client, collection, points, *, attempts=3, sleeper=None):
    """Retry only transient vector transport failures; stable IDs make replay idempotent."""
    import time
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    if attempts < 1:raise ValueError('At least one upsert attempt is required')
    sleeper = sleeper or time.sleep
    for attempt in range(attempts):
        try:
            return client.upsert(collection, points, wait=True)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            if isinstance(exc, UnexpectedResponse) and exc.status_code not in {408,429,500,502,503,504}:
                raise
            if attempt + 1 == attempts:raise
            print('VECTOR_BATCH_RETRY', attempt + 1, type(exc).__name__, flush=True)
            sleeper(2 ** (attempt + 1))

def restore(bundle):
    import numpy as np
    from sqlalchemy import select,func
    from qdrant_client import QdrantClient,models as qm
    app_imports()
    from app.db.tables import Base,DocumentBlock,DocumentImportRun
    from app.services import error_evidence_reviews  # noqa: F401
    from app.db.postgres import get_engine
    from app.config import get_settings
    m=verify(bundle);bundle=Path(bundle);data=bundle/'deployment/data';settings=get_settings()
    cfg=json.loads((data/'qdrant_config.json').read_text(encoding='utf-8'))
    inv=json.loads((data/'inventory.json').read_text(encoding='utf-8'))
    if cfg['embedding_provider']!=settings.embedding_provider:raise ValueError('Embedding provider mismatch')
    if settings.embedding_provider=='bge' and Path(settings.embedding_model).resolve()!=(bundle/'models/bge-small-zh-v1.5').resolve():raise ValueError('Use the bundled BGE model absolute path in .env')
    if inv['vectors']*cfg['size']*4!=(data/'vectors.f32').stat().st_size:raise ValueError('Vector matrix size mismatch')
    q=QdrantClient(url=settings.qdrant_url,timeout=120,check_compatibility=False);engine=get_engine()
    if q.collection_exists(settings.qdrant_collection):raise ValueError('Target collection exists; use a new empty demo instance')
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        for name in Base.metadata.tables:
            if c.execute(select(func.count()).select_from(Base.metadata.tables[name])).scalar():raise ValueError('Target database is not empty: '+name)
    # SQL transaction prevents partially restored PostgreSQL tables.
    with engine.begin() as c:
        for cls in [DocumentBlock,DocumentImportRun]:
            tab=cls.__table__;batch=[];n=0
            with gzip.open(data/(tab.name+'.jsonl.gz'),'rt',encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line)
                    if row['doc_id'] not in DOCS:raise ValueError('Foreign source in data archive')
                    for col in tab.columns:
                        if isinstance(row.get(col.name),str) and col.type.__class__.__name__=='DateTime':row[col.name]=datetime.fromisoformat(row[col.name])
                    batch.append(row)
                    if len(batch)==256:c.execute(tab.insert(),batch);n+=len(batch);batch=[]
                if batch:c.execute(tab.insert(),batch);n+=len(batch)
            print('restored',tab.name,n,flush=True)
    q.create_collection(settings.qdrant_collection,vectors_config=qm.VectorParams(size=cfg['size'],distance=qm.Distance(cfg['distance'])))
    count=0;batch=[]
    with gzip.open(data/'qdrant_points.jsonl.gz','rt',encoding='utf-8') as f,(data/'vectors.f32').open('rb') as v:
        for line in f:
            row=json.loads(line);raw=v.read(cfg['size']*4)
            if len(raw)!=cfg['size']*4:raise ValueError('Truncated matrix')
            batch.append(qm.PointStruct(id=row['id'],payload=row['payload'],vector=np.frombuffer(raw,dtype='<f4').tolist()))
            if len(batch)==256:
                upsert_batch(q,settings.qdrant_collection,batch);count+=len(batch);batch=[]
                if count%10240==0:print('restored vectors',count,flush=True)
        if batch:upsert_batch(q,settings.qdrant_collection,batch);count+=len(batch)
        if v.read(1):raise ValueError('Trailing vectors')
    assert count==inv['vectors'];q.close();engine.dispose();print('RESTORE_OK',count,flush=True)
def dependency_lock():
    from packaging.requirements import Requirement
    roots=['fastapi','uvicorn[standard]','pydantic','pydantic-settings','SQLAlchemy','psycopg[binary]','qdrant-client','langgraph','langchain-core','openai','PyMuPDF','opencv-python-headless','sentence-transformers','streamlit','pytest','httpx','python-dotenv','python-multipart']
    pending=[Requirement(r) for r in roots];found={};extras_seen={}
    while pending:
        req=pending.pop();dist=meta.distribution(req.name);key=dist.metadata['Name'].lower().replace('_','-');extras=set(req.extras)
        old=extras_seen.get(key)
        if old is not None and extras.issubset(old):continue
        extras_seen[key]=(old or set())|extras;found[key]=(dist.metadata['Name'],dist.version)
        for raw in dist.requires or []:
            r=Requirement(raw)
            if not r.marker or any(r.marker.evaluate({'extra':e}) for e in extras_seen[key]|{''}):pending.append(r)
    return '\n'.join(f'{name}=={version}' for name,version in sorted(found.values(),key=lambda a:a[0].lower()))+'\n'
def build(out,data,version,wheelhouse=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    # Explicit allowlist; never package .git, .env, runtime media or arbitrary untracked files.
    for dirname in ['backend','frontend','prompts','scripts','tests','evaluation','docs','simulation','.github','deployment']:
        src=ROOT/dirname
        if not src.exists():continue
        for p in src.rglob('*'):
            rel=p.relative_to(ROOT)
            if not p.is_file() or any(s.startswith('_') and s!='__init__.py' for s in rel.parts) or '__pycache__' in rel.parts:continue
            if dirname=='evaluation' and 'reports' in rel.parts and not p.name.startswith(('latest-','candidate-latest-')):continue
            if p.suffix.lower() not in {'.py','.md','.json','.yml','.yaml','.ps1','.cmd','.toml','.ini','.txt','.svg','.png','.jpg','.jpeg','.html','.js','.css'}:continue
            dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for name in ['requirements.txt','pytest.ini','启动学生端.cmd','启动题目导入.cmd','启动错误复核.cmd']:
        shutil.copy2(ROOT/name,out/name)
    shutil.copytree(ROOT/'data/problems',out/'data/problems')
    shutil.copy2(ROOT/'data/knowledge_vocabulary.json',out/'data/knowledge_vocabulary.json')
    shutil.copytree(ROOT/'data/documents',out/'data/documents')
    shutil.copytree(data,out/'deployment/data')
    cache=Path.home()/'.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots'
    snaps=sorted(cache.iterdir())
    if not snaps:raise ValueError('BGE model cache missing')
    model_src=snaps[-1];shutil.copytree(model_src,out/'models/bge-small-zh-v1.5',symlinks=False)
    (out/'requirements-deploy.lock.txt').write_text(dependency_lock(),encoding='utf-8')
    (out/'deployment/runtime-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True),encoding='utf-8')
    shutil.copy2(ROOT/'deployment/.env.deploy.example',out/'.env.example')
    shutil.copy2(ROOT/'deployment/compose.yml',out/'docker-compose.yml')
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    version_info={'version':version,'base_git_commit':git_head,'source':'current working tree, includes uncommitted bytes','created_at':datetime.now().isoformat(),'python':platform.python_version(),'platform':platform.platform(),'embedding_revision':model_src.name,'source_validation':'See deployment/验收记录.md; no automatic claim that historical v16 covers packaging changes'}
    write(out/'VERSION.json',version_info)
    inventory=json.loads((out/'deployment/data/inventory.json').read_text(encoding='utf-8'))
    inventory.update({'figures':sum(1 for p in (out/'data/documents').rglob('*') if p.is_file()),'problems':sorted(p.name for p in (out/'data/problems').iterdir() if p.is_dir()),'raw_textbook_pdfs':'not included; parsed blocks/vectors/figures included','student_history':'not included','api_keys':'not included','wheels':'not included; initial pip install needs network','docker_images':'not included; initial image pull needs network','ranking_note':'Vectors and payloads preserved; restored HNSW index rebuild may change near-tie ranks; do not claim bit-identical retrieval results'})
    write(out/'DATA_INVENTORY.json',inventory)
    if wheelhouse:
        provenance=json.loads((ROOT/'deployment/wheel-provenance.json').read_text(encoding='utf-8'))
        dest=out/'wheels';dest.mkdir()
        for name,entry in provenance['files'].items():
            p=Path(wheelhouse)/name
            if sha(p)!=entry['sha256']:raise ValueError('Wheel identity mismatch: '+name)
            shutil.copy2(p,dest/name)
        inventory['wheels']={'count':len(provenance['files']),'verification':'official PyPI SHA256','platform':provenance['platform']}
        write(out/'DATA_INVENTORY.json',inventory)
    files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'manifest.json',{'schema':'shiwu-deploy.v1','version':version,'files':files,'excluded':['.env','.venv','.git','student database tables','video jobs/runs','logs','original videos','raw textbook PDFs','artifacts','output presentations','temporary scripts']})
    verify(out);print('BUNDLE_BUILT',out,flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('export-data');a.add_argument('--out',required=True)
    a=s.add_parser('build');a.add_argument('--out',required=True);a.add_argument('--data',required=True);a.add_argument('--version',required=True);a.add_argument('--wheelhouse')
    for cmd in ['verify','restore']:
        a=s.add_parser(cmd);a.add_argument('--bundle',default=str(ROOT))
    a=p.parse_args()
    if a.command=='export-data':export_data(a.out)
    elif a.command=='build':build(a.out,a.data,a.version,a.wheelhouse)
    elif a.command=='verify':verify(a.bundle)
    else:restore(a.bundle)
if __name__=='__main__':main()
