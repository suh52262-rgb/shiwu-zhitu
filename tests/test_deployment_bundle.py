"""Packaging guardrails; no real services or model requests."""
import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deploy_bundle',ROOT/'scripts/deployment_bundle.py')
bundle=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundle)
def test_student_tables_are_not_exportable():
    assert bundle.TABLES==['document_blocks','document_import_runs']
    assert len(bundle.DOCS)==8
@pytest.mark.parametrize('value',['figures/p0001.png','data/documents/jiaocai-1-jieda/figures/p0001.png'])
def test_relative_asset_paths_keep_identity(value):assert bundle.portable_figure(value)==value
def test_absolute_asset_path_becomes_portable():
    assert bundle.portable_figure(str(Path(__file__).resolve().parents[1] / 'data/documents/doc/figures/a.png'))=='data/documents/doc/figures/a.png'
@pytest.mark.parametrize('value',['figures/../secret','C:/secret/key','data/documents/../../secret'])
def test_asset_traversal_and_foreign_paths_rejected(value):
    with pytest.raises(ValueError):bundle.portable_figure(value)
def test_hash_verification_fails_closed(tmp_path):
    p=tmp_path/'file.txt';p.write_text('original',encoding='utf-8')
    bundle.write(tmp_path/'manifest.json',{'files':{'file.txt':{'bytes':p.stat().st_size,'sha256':bundle.sha(p)}}})
    bundle.verify(tmp_path);p.write_text('modified',encoding='utf-8')
    with pytest.raises(ValueError):bundle.verify(tmp_path)
def test_manifest_cannot_reference_parent(tmp_path):
    bundle.write(tmp_path/'manifest.json',{'files':{'../outside':{'bytes':0,'sha256':''}}})
    with pytest.raises(ValueError):bundle.verify(tmp_path)

def test_vector_restore_retries_same_ids_only_on_transient_failure():
    from qdrant_client.http.exceptions import ResponseHandlingException
    points=[{'id':'stable'}];calls=[];delays=[]
    class Client:
        def upsert(self,collection,items,wait):
            calls.append((collection,items,wait))
            if len(calls)<3:raise ResponseHandlingException(ConnectionError('transient'))
            return 'ok'
    assert bundle.upsert_batch(Client(),'test',points,sleeper=delays.append)=='ok'
    assert len(calls)==3 and all(c[1] is points for c in calls) and delays==[2,4]

def test_vector_restore_exhaustion_is_not_reported_success():
    from qdrant_client.http.exceptions import ResponseHandlingException
    class Client:
        def upsert(self,*args,**kwargs):raise ResponseHandlingException(ConnectionError('persistent'))
    with pytest.raises(ResponseHandlingException):bundle.upsert_batch(Client(),'test',[],sleeper=lambda _:None)

def test_vector_restore_validation_error_is_not_retried():
    calls=[]
    class Client:
        def upsert(self,*args,**kwargs):calls.append(1);raise ValueError('invalid vector')
    with pytest.raises(ValueError):bundle.upsert_batch(Client(),'test',[],sleeper=lambda _:None)
    assert calls==[1]
