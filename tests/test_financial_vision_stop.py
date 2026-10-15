import types,sys
import pytest
from app.services import vision


def test_failed_account_does_not_try_second_vision_model(monkeypatch,tmp_path):
    monkeypatch.setenv('LLM_CIRCUIT_DIR',str(tmp_path))
    monkeypatch.setenv('DEEPSEEK_API_KEY','isolated-fake-key')
    monkeypatch.setattr(vision,'vision_candidates',lambda:[vision.Candidate('deepseek','one'),vision.Candidate('deepseek','two')])
    calls=[]
    class BalanceError(Exception):status_code=402
    class FakeSDK:
        def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=self)
        def create(self,**kwargs):calls.append(kwargs['model']);raise BalanceError('test balance')
    monkeypatch.setitem(sys.modules,'openai',types.SimpleNamespace(OpenAI=FakeSDK))
    with pytest.raises(vision.VisionUnavailable):vision.call_vision(b'fake image','read it')
    assert calls==['one']
