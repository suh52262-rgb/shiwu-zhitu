"""Run real_db tests only against newly created disposable PostgreSQL/Qdrant."""
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from accept_real_videos import disposable_services


def main(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a fresh evidence directory')
    out.mkdir(parents=True)
    with disposable_services():
        env=dict(os.environ);env.pop('TEST_OFFLINE',None)
        env['CI_ISOLATED_SERVICES']='1';env['PYTHONPATH']=str(ROOT/'backend')
        setup='from app.db.postgres import get_engine; from app.db.tables import Base; import app.services.error_evidence_reviews; Base.metadata.create_all(get_engine())'
        subprocess.run([sys.executable,'-c',setup],cwd=ROOT,env=env,check=True)
        with (out/'pytest.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-m','pytest','-q','-m','real_db',
                                   '--junitxml='+str(out/'pytest.xml')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Isolated service group failed; see saved log')
        print('PASS disposable PostgreSQL/Qdrant real_db tests')


if __name__=='__main__':main(sys.argv[1])
