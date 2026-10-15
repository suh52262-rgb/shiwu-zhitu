"""Isolated disposable PostgreSQL API acceptance; never touches the source database."""
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))


def main(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True)
    name='shiwu-error-review-test-'+uuid4().hex[:10]
    def docker(*args):return subprocess.check_output(['docker',*args],text=True).strip()
    docker('run','-d','--name',name,'-e','POSTGRES_USER=acceptance',
           '-e','POSTGRES_PASSWORD=isolated_acceptance','-e','POSTGRES_DB=acceptance',
           '-p','127.0.0.1::5432','postgres:16-alpine')
    engine=None
    try:
        port=docker('port',name,'5432/tcp').rsplit(':',1)[-1]
        from sqlalchemy import create_engine,text
        from sqlalchemy.orm import Session
        from app.db.tables import Base,Student,Problem,Attempt
        from app.services import error_evidence_reviews as service
        from app.db import persistence
        from app.services.learning_profile import evidence_id
        from app.api import error_evidence_reviews as api
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        engine=create_engine(f'postgresql+psycopg://acceptance:isolated_acceptance@127.0.0.1:{port}/acceptance')
        for _ in range(40):
            try:
                with engine.connect() as connection:connection.execute(text('select 1'))
                break
            except Exception:time.sleep(.5)
        else:raise RuntimeError('Isolated PostgreSQL did not become ready')
        Base.metadata.create_all(engine)
        service.get_engine=lambda:engine
        persistence.get_engine=lambda:engine
        evidence={'concept':'牛顿第二定律','issue':'合力写成14 N','student_text':'F=10+4=14 N',
                  'source_id':'fixture_photo','source_revision':0}
        state={'profile_policy_version':'paper_evidence_v1','error_evidence':[evidence]}
        with Session(engine) as session,session.begin():
            session.add_all([Student(id='isolated_ai_student'),Problem(id='isolated_ai_problem',content='AI工程测试题')])
            session.flush()
            session.add(Attempt(id='isolated_ai_attempt',student_id='isolated_ai_student',problem_id='isolated_ai_problem',state=state))
        app=FastAPI();app.include_router(api.router)
        client=TestClient(app);base='/teacher/students/isolated_ai_student/error-evidence'
        before=client.get(base);assert before.status_code==200
        eid=evidence_id('isolated_ai_attempt',evidence)
        body={'attempt_id':'isolated_ai_attempt','decision':'resolved','actor_label':'AI隔离工程验收',
              'note':'仅验证操作记录，不证明学生掌握','client_request_id':str(uuid4())}
        first=client.post(base+'/'+eid+'/decisions',json=body);assert first.status_code==200
        duplicate=client.post(base+'/'+eid+'/decisions',json=body)
        assert duplicate.status_code==200 and first.json()==duplicate.json()
        conflict=client.post(base+'/'+eid+'/decisions',json={**body,'decision':'dismissed'})
        assert conflict.status_code==409
        cross=client.post(base.replace('isolated_ai_student','other')+'/'+eid+'/decisions',json=body)
        assert cross.status_code==404
        profile=persistence.get_student_profile('isolated_ai_student')
        assert profile['observed_error_concepts']==[] and len(profile['error_evidence'])==1
        with Session(engine) as session:
            assert session.get(Attempt,'isolated_ai_attempt').state==state
            assert session.query(service.ErrorEvidenceReview).count()==1
        changed={**state,'error_evidence':[{**evidence,'source_revision':1,'student_text':'F=10-4=6 N'}]}
        with Session(engine) as session,session.begin():session.get(Attempt,'isolated_ai_attempt').state=changed
        stale=client.post(base+'/'+eid+'/decisions',json={**body,'client_request_id':str(uuid4())})
        assert stale.status_code==409
        after=client.get(base);assert after.status_code==200
        assert after.json()['items'][0]['review_status']=='unreviewed'
        record={'passed':True,'scope':'Isolated PostgreSQL + real FastAPI router + persistence; AI fixtures; no model calls',
                'before':before.json(),'decision':first.json(),'after_revision':after.json(),
                'http_statuses':{'create':200,'duplicate':200,'conflict':409,'cross_student':404,'stale_revision':409},
                'student_state_unchanged_by_decision':True,'append_only_row_count':1}
        (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS isolated PostgreSQL error-evidence API acceptance')
    finally:
        if engine:engine.dispose()
        docker('rm','-f',name)


if __name__=='__main__':main(sys.argv[1])
