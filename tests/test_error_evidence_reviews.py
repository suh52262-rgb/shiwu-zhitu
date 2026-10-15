from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from app.db.tables import Base,Student,Problem as ProblemRow,Attempt
from app.services import error_evidence_reviews as reviews
from app.services.learning_profile import profile_signals,evidence_id
from sqlalchemy.orm import Session


@pytest.fixture
def setup_review(tmp_path,monkeypatch):
    engine=create_engine('sqlite:///'+str(tmp_path/'reviews.sqlite'))
    Base.metadata.create_all(engine)
    monkeypatch.setattr(reviews,'get_engine',lambda:engine)
    evidence={'concept':'牛顿第二定律','issue':'合力写成14 N','student_text':'F=10+4=14 N','source_id':'photo_1','source_revision':0}
    state={'profile_policy_version':'paper_evidence_v1','error_evidence':[evidence]}
    with Session(engine) as session,session.begin():
        session.add_all([Student(id='s'),ProblemRow(id='p',content='题')]);session.flush()
        session.add(Attempt(id='a',student_id='s',problem_id='p',state=state))
    return engine,evidence,state


def test_resolution_changes_active_recommendation_signals_without_deleting_evidence(setup_review):
    _,e,state=setup_review;eid=evidence_id('a',e);rid=str(uuid4())
    saved=reviews.append('s','a',eid,'resolved','AI工程测试（隔离）','有新的正确书写供核对',rid)
    assert reviews.append('s','a',eid,'resolved','AI工程测试（隔离）','有新的正确书写供核对',rid)==saved
    assert len(reviews.decisions('s'))==1
    signals=profile_signals([('a',state)],decisions=reviews.decisions('s'))
    assert signals['observed_error_concepts']==[]
    assert len(signals['error_evidence'])==1 and signals['error_evidence'][0]['review_status']=='resolved'
    assert not saved['semantic_validated']


def test_new_revision_does_not_inherit_old_review_and_stale_action_rejected(setup_review):
    engine,e,state=setup_review;eid=evidence_id('a',e)
    reviews.append('s','a',eid,'dismissed','AI测试','误读',str(uuid4()))
    new={**e,'source_revision':1,'student_text':'F=10-4=6 N'}
    changed={**state,'error_evidence':[new]}
    assert profile_signals([('a',changed)],decisions=reviews.decisions('s'))['error_evidence'][0]['review_status']=='unreviewed'
    with Session(engine) as session,session.begin():session.get(Attempt,'a').state=changed
    with pytest.raises(LookupError):reviews.append('s','a',eid,'confirmed','AI测试','旧修订',str(uuid4()))


def test_request_conflict_and_cross_student_do_not_overwrite_history(setup_review):
    _,e,_=setup_review;eid=evidence_id('a',e);rid=str(uuid4())
    reviews.append('s','a',eid,'confirmed','AI测试','核对',rid)
    with pytest.raises(ValueError):reviews.append('s','a',eid,'dismissed','AI测试','核对',rid)
    with pytest.raises(FileNotFoundError):reviews.append('other','a',eid,'resolved','AI测试','核对',str(uuid4()))
    assert len(reviews.decisions('s'))==1
