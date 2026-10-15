from io import BytesIO
import threading
from uuid import uuid4
import pytest
from fastapi import UploadFile, HTTPException
from app.services.photo_jobs import PhotoJobQueue
from app.services.video_progress import progress_callback


def upload():return UploadFile(file=BytesIO(b'photo bytes'),filename='photo.png')


def test_partial_recognition_idempotency_and_restart(tmp_path):
    recognized=threading.Event();resume=threading.Event();calls=[]
    def runner(file, **params):
        calls.append(params)
        progress_callback.get()('teaching',{'recognized_text':'F=ma','completed':1,'total':1,'unit':'images'})
        recognized.set();assert resume.wait(5)
        return {'attempt_id':'a','upload_removed':True}
    q=PhotoJobQueue(tmp_path,runner=runner)
    params=dict(student_id='s',problem_id='p',attempt_id='',request_hint=True,include_review=False)
    rid=str(uuid4());job=q.submit(upload(),params,100,rid)
    assert recognized.wait(5)
    d=q.get(job['job_id'],'s','p');assert d['recognized_text']=='F=ma' and d['status']=='running'
    assert q.submit(upload(),params,100,rid)['job_id']==job['job_id']
    with pytest.raises(HTTPException) as wrong:q.get(job['job_id'],'other','p')
    assert wrong.value.status_code==404
    recovered=PhotoJobQueue(tmp_path,runner=lambda *_:pytest.fail('must not rerun'))
    assert recovered.get_by_request(rid,'s','p')['status']=='interrupted'
    resume.set();q.pending.join()
    assert len(calls)==1 and recovered.get(job['job_id'],'s','p')['status']=='interrupted'

def test_new_photo_binds_attempt_before_ocr_and_replay_compares_original_params(tmp_path, monkeypatch):
    from app.api import attempts, video
    from app.services.problem_repository import get_problem_repository
    q=PhotoJobQueue(tmp_path)
    params=dict(student_id='photo_s',problem_id='demo_02',attempt_id='',request_hint=True,include_review=False)
    bound=q.prepare_params(params)
    assert bound['attempt_id'] and bound['_requested_attempt_id']==''
    assert q._same_params({'params':bound},params)
    assert not q._same_params({'params':bound},{**params,'request_hint':False})
    record=attempts.get_attempt_store().load_attempt(bound['attempt_id'])
    assert record and not record.state.timeline
    attempts.get_attempt_store().abandon_attempt(bound['attempt_id'])
    with pytest.raises(HTTPException) as error:
        video._validate_media_target('photo_s','demo_02',bound['attempt_id'])
    assert error.value.status_code==409


def test_same_photo_request_rejects_changed_bytes_and_params(tmp_path):
    q=PhotoJobQueue(tmp_path,runner=lambda *args,**kwargs:{'upload_removed':True})
    params=dict(student_id='s',problem_id='p',attempt_id='a',request_hint=True,include_review=False)
    rid=str(uuid4());q.submit(upload(),params,100,rid)
    with pytest.raises(HTTPException) as error:
        q.submit(UploadFile(file=BytesIO(b'changed'),filename='photo.png'),params,100,rid)
    assert error.value.status_code==409
    with pytest.raises(HTTPException) as error:
        q.submit(upload(),{**params,'request_hint':False},100,rid)
    assert error.value.status_code==409
    q.pending.join()

def test_photo_api_rejects_blank_request_id_before_any_queue_or_model(monkeypatch):
    from app.api import photo_jobs
    monkeypatch.setattr(photo_jobs,'get_photo_jobs',lambda:pytest.fail('must reject before queuing'))
    with pytest.raises(HTTPException) as error:
        photo_jobs.create(file=upload(),student_id='s',problem_id='p',attempt_id='',request_hint=True,client_request_id='   ')
    assert error.value.status_code==422
