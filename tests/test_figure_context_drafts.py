import json
from uuid import uuid4
import pytest
from app.models.problem import Problem
from app.services.problem_repository import ProblemRepository
from app.services.graph_versions import atomic_json
from app.services.figure_context_drafts import FigureContextDrafts
from app.services.problem_import import ImportConflict


def make_service(tmp_path):
    repo=ProblemRepository(tmp_path/'bank')
    figure=repo.problem_dir('p')/'figures/one.png';figure.parent.mkdir(parents=True);figure.write_bytes(b'image-content')
    atomic_json(repo.problem_path('p'),Problem(problem_id='p',statement='2kg物体受力',figures=['figures/one.png'],metadata={'source':'reviewed_import'}).model_dump())
    calls=[]
    def vision(*args,**kwargs):
        calls.append(kwargs)
        return json.dumps({'description':'物块m=2kg，箭头F=8N向右','uncertainties':['是否光滑未标注']},ensure_ascii=False),{'model':'fake'}
    return FigureContextDrafts(tmp_path/'drafts',repo,vision),figure,calls


def test_repeat_is_cached_and_input_changes_are_conflicts(tmp_path):
    service,figure,calls=make_service(tmp_path);rid=str(uuid4())
    first=service.submit('p',rid);second=service.submit('p',rid)
    assert first['status']=='ready' and first==second and len(calls)==1
    assert first['review_required'] and not first['semantic_validated']
    assert 'raw_response' not in first
    figure.write_bytes(b'changed-image')
    assert service.get('p',rid)['stale']
    with pytest.raises(ImportConflict):service.submit('p',rid)
    assert len(calls)==1


def test_unknown_figure_and_cross_problem_access_never_call_model(tmp_path):
    service,_,calls=make_service(tmp_path)
    with pytest.raises(ValueError):service.submit('p',str(uuid4()),9)
    rid=str(uuid4());service.submit('p',rid)
    with pytest.raises(FileNotFoundError):service.get('other',rid)
    assert len(calls)==1


def test_restart_processing_record_is_interrupted_without_paid_retry(tmp_path):
    service,_,calls=make_service(tmp_path);rid=str(uuid4())
    record=service.submit('p',rid)
    stored=json.loads(service.path(rid).read_text(encoding='utf-8'));stored.update(status='processing',owner_epoch='old-process')
    atomic_json(service.path(rid),stored)
    assert service.get('p',rid)['status']=='interrupted'
    assert service.submit('p',rid)['status']=='interrupted'
    assert len(calls)==1


def test_recovery_and_duplicate_submission_do_not_wait_for_vision(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    service,_,calls=make_service(tmp_path);rid=str(uuid4())
    entered=Event();release=Event()
    original=service.vision
    def blocked(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    service.vision=blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(service.submit,'p',rid)
        try:
            assert entered.wait(3)
            assert pool.submit(service.get,'p',rid).result(timeout=1)['status']=='processing'
            assert pool.submit(service.submit,'p',rid).result(timeout=1)['status']=='processing'
        finally:release.set()
        assert first.result(timeout=3)['status']=='ready'
    assert len(calls)==1
