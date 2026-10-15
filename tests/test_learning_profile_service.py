import uuid
import pytest
from sqlalchemy import text


@pytest.mark.real_db
def test_current_evidence_and_legacy_help_are_separate_in_postgres():
    from app.db import persistence
    from app.db.postgres import get_engine
    from app.models.problem import Problem
    from app.models.student import StudentState
    from app.services.attempt_store import AttemptStore
    from app.api.teacher import teacher_knowledge_stats

    suffix=uuid.uuid4().hex[:10]
    sid,pid='profile_'+suffix,'profile_question_'+suffix
    concept='profile_error_'+suffix
    store=AttemptStore(persistent=True)
    try:
        record=store.create_attempt(sid,pid,Problem(problem_id=pid,statement='test input'))
        assert persistence.persist_profile_weak_concepts(sid,['historical_help_only'])
        state=StudentState(student_id=sid,problem_id=pid,profile_policy_version='paper_evidence_v1',
            help_events=[{'level':5}],error_evidence=[{'concept':concept,'issue':'observed relation',
            'student_text':'F=0','semantic_validated':False}])
        store.save_attempt(record,state)
        profile=persistence.get_student_profile(sid)
        assert profile['observed_error_concepts']==[concept]
        assert profile['legacy_weak_concepts']==['historical_help_only']
        assert len(profile['help_events'])==1
        stats=teacher_knowledge_stats(limit=1000)
        assert any(r['knowledge_point']==concept for r in stats['observed_error_knowledge_points'])
    finally:
        with get_engine().begin() as connection:
            for table in ('attempts','student_profiles','students'):
                key='id' if table=='students' else 'student_id'
                connection.execute(text(f'DELETE FROM {table} WHERE {key}=:sid'),{'sid':sid})
            connection.execute(text('DELETE FROM problems WHERE id=:pid'),{'pid':pid})
