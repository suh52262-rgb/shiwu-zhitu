import pytest
from app.services import request_execution as execution
from app.services.llm.base import LLMTimeoutError


def test_deadline_never_extends_and_restores(monkeypatch):
    now=[10.0];monkeypatch.setattr(execution.time,'monotonic',lambda:now[0])
    with execution.request_execution(5,roles={'tutor':{'thinking':'disabled'}}):
        assert execution.bounded_timeout(30)==5
        with execution.request_execution(90):assert execution.bounded_timeout(30)==5
        now[0]=16
        with pytest.raises(LLMTimeoutError):execution.bounded_timeout(30)
        assert execution.role_options('tutor')['thinking']=='disabled'
    assert execution.remaining_seconds() is None and execution.role_options('tutor')=={}

def test_expired_budget_does_not_start_retry_and_records_failed_attempt(monkeypatch):
    from app.services.llm.base import LLMProvider
    from app.services.llm.tracking import track_usage
    now=[10.0];monkeypatch.setattr(execution.time,'monotonic',lambda:now[0])
    class SlowProvider(LLMProvider):
        def _complete(self,messages):
            now[0]=16.0
            raise LLMTimeoutError('slow upstream')
    with track_usage() as usage, execution.request_execution(5):
        with pytest.raises(LLMTimeoutError):SlowProvider(model='fake',max_retries=5).generate_text([])
        assert usage.as_dict()['actual_attempts']==1
        assert usage.as_dict()['failed_attempts']==['LLMTimeoutError']
