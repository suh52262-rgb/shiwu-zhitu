"""Exercise the actual read-only replay entrypoint, not a copied assembly rule."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def setup_replay(tmp_path):
    dbpath=tmp_path/'corpus.sqlite'
    with sqlite3.connect(dbpath) as db:
        db.execute('create table document_blocks(id text,doc_id text,page integer,type text,text text,indexable integer,is_generated integer,bbox text,block_index integer)')
        db.executemany('insert into document_blocks values(?,?,?,?,?,?,?,?,?)',[
            ('a','book',1,'text','式（1.2）说明',1,0,'[]',0),
            ('f','book',1,'formula',r'x=y\tag{1.2}',1,0,'[]',1)])
    audit=tmp_path/'audit';audit.mkdir()
    # Replay consumes already-verified sidecars. This fixture does not pretend to
    # be a PDF/semantic audit; original-span verification has separate tests.
    original=tmp_path/'original.source';original.write_bytes(b'fixed original source')
    write(audit/'protocol.json',{'database_sha256':hashlib.sha256(dbpath.read_bytes()).hexdigest(),
        'pdfs':{'book':{'path':str(original),'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}}})
    write(audit/'checks.json',[{'formula_id':'f','source':'book','page':1,'reference':'1.2',
        'label_id':'label','result':{'status':'verified'}}])
    inputs=tmp_path/'inputs';inputs.mkdir()
    write(inputs/'protocol.json',{'dataset':{'cases':[{'id':'case'}]}})
    write(inputs/'case.json',{'id':'case','joint_pool':[{'id':'a','source':'book','page':1,
        'type':'text','text':'式（1.2）说明'}]})
    return dbpath,audit,inputs,original


def execute(tmp_path,db,audit,inputs):
    return subprocess.run([sys.executable,str(ROOT/'scripts/replay_verified_label_candidates.py'),
        '--database',str(db),'--audit',str(audit),'--inputs',str(inputs),
        '--out',str(tmp_path/'out')],capture_output=True,text=True,timeout=30)


def test_actual_replay_preserves_corpus_and_copies_original_formula(tmp_path):
    db,audit,inputs,_=setup_replay(tmp_path);before=db.read_bytes()
    run=execute(tmp_path,db,audit,inputs)
    assert run.returncode==0,run.stderr
    result=json.loads((tmp_path/'out/case.json').read_text(encoding='utf-8'))
    assert [b['id'] for b in result['added']]==['f']
    assert result['added'][0]['text']==r'x=y\tag{1.2}'
    assert not result['semantic_sufficiency_evaluated']
    assert db.read_bytes()==before


@pytest.mark.parametrize('change',['missing_case','missing_pool','wrong_text','source_changed','corpus_changed'])
def test_replay_refuses_incomplete_or_changed_sources(tmp_path,change):
    db,audit,inputs,original=setup_replay(tmp_path)
    if change=='missing_case':(inputs/'case.json').unlink()
    elif change=='missing_pool':write(inputs/'case.json',{'id':'case'})
    elif change=='wrong_text':
        data=json.loads((inputs/'case.json').read_text(encoding='utf-8'))
        data['joint_pool'][0]['text']='虚构式（1.2）';write(inputs/'case.json',data)
    elif change=='source_changed':original.write_bytes(b'changed')
    else:
        with sqlite3.connect(db) as conn:conn.execute("update document_blocks set text='changed' where id='a'")
    run=execute(tmp_path,db,audit,inputs)
    assert run.returncode!=0
    reason={'missing_case':'FileNotFoundError','missing_pool':'Frozen case missing candidate pool',
            'wrong_text':'Invalid original pool identity/text','source_changed':'Original PDF changed',
            'corpus_changed':'Corpus differs from printed-label audit'}[change]
    assert reason in run.stderr
    assert not (tmp_path/'out/summary.json').exists()
