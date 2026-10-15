from copy import deepcopy
from datetime import datetime, date
from io import BytesIO
import json

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.teacher_evidence import router
from app.db.tables import Base, Attempt, Student, Problem
from app.services import teacher_read as read
from app.services.learning_profile import evidence_id
from app.services.error_evidence_reviews import ErrorEvidenceReview


@pytest.fixture
def database(monkeypatch):
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(read, 'get_engine', lambda: engine)
    with Session(engine) as session:
        session.add_all([Student(id='s'), Problem(id='p')])
        session.add(Attempt(id='a', student_id='s', problem_id='p', start_time=datetime(2026, 10, 14, 12),
            state={'profile_policy_version': 'paper_evidence_v1', 'error_evidence': [
                {'concept': 'force', 'issue': 'wrong equation', 'student_text': 'F=2', 'source_id': 'source', 'source_revision': 0}]}))
        session.commit()
    yield engine
    engine.dispose()


def test_empty_scope_is_not_all_students_and_dates_are_inclusive(database, monkeypatch):
    monkeypatch.setattr(read, 'read_decisions', lambda: ([], 'ready'))
    assert read.scoped_errors(student_ids=[])['attempt_count'] == 0
    assert read.scoped_errors(start=date(2026, 10, 14), end=date(2026, 10, 14))['attempt_count'] == 1
    assert read.scoped_errors(end=date(2026, 10, 13))['attempt_count'] == 0


@pytest.mark.parametrize('decision,active', [('confirmed', True), ('dismissed', False), ('resolved', False)])
def test_current_evidence_obeys_latest_local_decision(database, monkeypatch, decision, active):
    with Session(database) as session:
        item = session.get(Attempt, 'a').state['error_evidence'][0]
    eid = evidence_id('a', item)
    monkeypatch.setattr(read, 'read_decisions', lambda: ([{'evidence_id': eid, 'decision': decision}], 'ready'))
    result = read.scoped_errors()
    assert result['evidence'][0]['active'] is active
    assert result['counts'][decision] == 1


def test_unavailable_review_store_is_not_zero_error_count(database, monkeypatch):
    monkeypatch.setattr(read, 'read_decisions', lambda: ([], 'unavailable'))
    result = read.scoped_errors()
    assert result['counts'] is None
    assert result['evidence'][0]['review_status'] == 'unavailable'
    assert not result['evidence'][0]['active']


def test_image_endpoint_rejects_wrong_attempt_owner(database, monkeypatch, tmp_path):
    monkeypatch.setattr(read, 'RUNS', tmp_path/'runs')
    monkeypatch.setattr(read, 'PHOTOS', tmp_path/'photos')
    app = FastAPI(); app.include_router(router)
    client = TestClient(app)
    assert client.get('/teacher/attempts/a/evidence', params={'student_id':'other','problem_id':'p'}).status_code == 404
    assert client.get('/teacher/attempts/a/images/arbitrary', params={'student_id':'s','problem_id':'p'}).status_code == 404


def test_new_photo_is_retained_verbatim_and_only_in_own_catalogue(monkeypatch, tmp_path):
    monkeypatch.setattr(read, 'PHOTOS', tmp_path/'photos')
    monkeypatch.setattr(read, 'RUNS', tmp_path/'runs')
    raw = BytesIO(); Image.new('RGB', (2, 2)).save(raw, format='PNG')
    identity = 'f'*32
    assert read.retain_photo(raw.getvalue(), source_id=identity, student_id='s', problem_id='p', attempt_id='a') == 'retained'
    context = {'attempt_id':'a','student_id':'s','problem_id':'p','state':{'media_sources':[{'source_id':identity}]}}
    images = read.image_catalogue(context)
    assert len(images) == 1 and images[0]['_path'].read_bytes() == raw.getvalue()
    assert read.image_catalogue({**context, 'student_id':'other'}) == []
    assert read.retain_photo(b'not an image', source_id='e'*32, student_id='s',problem_id='p',attempt_id='a') == 'failed'


def test_material_revision_or_continuation_invalidates_report():
    state = {'media_sources':[{'source_id':'one','revision':2},{'source_id':'two','revision':1}], 'timeline':[{'event_type':'answer'}]}
    material = {'source_revisions':{'one':2,'two':1}, 'media_source_ids':['one','two'],
                'timeline_length':1, 'timeline_snapshot':deepcopy(state['timeline'])}
    assert read.current_material(material, state)
    changed=deepcopy(state); changed['media_sources'][1]['revision']=2
    assert not read.current_material(material, changed)
    changed=deepcopy(state); changed['timeline'].append({'event_type':'answer'})
    assert not read.current_material(material, changed)


def test_original_frame_directory_rejects_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(read,'RUNS',tmp_path/'runs');monkeypatch.setattr(read,'PHOTOS',tmp_path/'photos')
    directory=read.RUNS/('a'*32);directory.mkdir(parents=True)
    context={'attempt_id':'a','student_id':'s','problem_id':'p','state':{}}
    material={k:context[k] for k in ['attempt_id','student_id','problem_id']}
    (directory/'review_input.json').write_text(json.dumps({'input':material,'fingerprint':read.fingerprint(material)}))
    (directory/'results.json').write_text(json.dumps({'frames':[{'frame':'../secret.jpg'}]}))
    (directory/'secret.jpg').write_bytes(b'secret')
    assert read.image_catalogue(context)==[]


def test_read_decisions_never_creates_store(database, monkeypatch):
    from app.services import error_evidence_reviews
    def forbidden():raise AssertionError('DDL forbidden in teacher GET')
    monkeypatch.setattr(error_evidence_reviews,'ensure_store',forbidden)
    assert read.read_decisions()[1]=='ready'


def test_report_reader_keeps_current_and_superseded_material_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(read,'RUNS',tmp_path/'runs')
    directory=read.RUNS/('a'*32);(directory/'review_reports').mkdir(parents=True)
    context={'attempt_id':'a','student_id':'s','problem_id':'p','state':{'media_sources':[{'source_id':'one','revision':0}], 'timeline':[]}}
    material={k:context[k] for k in ['attempt_id','student_id','problem_id']}
    material.update(source_revisions={'one':0},timeline_length=0,timeline_snapshot=[],media_source_ids=['one'])
    identity=read.fingerprint(material)
    (directory/'review_input.json').write_text(json.dumps({'input':material,'fingerprint':identity}))
    payload={'timeline':'stored report','sticking_points':[],'key_concept':'force','next_step':'check','encouragement':'continue','answer_check':None,'render_policy':'legacy','canonicalized_fields':[]}
    (directory/'review_reports/report.json').write_text(json.dumps({'input_revision':0,'review_input_fingerprint':identity,'review':payload,'generation_fingerprint':'generation'}))
    from app.services import review_builder
    monkeypatch.setattr(review_builder,'review_trajectory',lambda *a,**k:pytest.fail('Teacher read must not generate report'))
    assert read.reports(context)['reports'][0]['material_status']=='current'
    context['state']['media_sources'][0]['revision']=1
    assert read.reports(context)['reports'][0]['material_status']=='superseded'
    changed=json.loads((directory/'review_reports/report.json').read_text());changed['review_input_fingerprint']='wrong'
    (directory/'review_reports/report.json').write_text(json.dumps(changed))
    result=read.reports(context)
    assert result['reports']==[] and result['issues']
