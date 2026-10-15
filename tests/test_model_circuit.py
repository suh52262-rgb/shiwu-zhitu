import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.services.llm.circuit import AccountCircuit
from app.services.llm.base import LLMResponseError,LLMConfigError,LLMProvider
from app.services.llm.schemas import TokenUsage


class Fake(LLMProvider):
    def __init__(self,guard,code):
        super().__init__('fake',max_retries=2);self.request_guard=guard;self.code=code;self.calls=0
    def _complete(self,messages):
        self.calls+=1
        if self.code:raise LLMResponseError('safe test',self.code)
        return 'ok',TokenUsage()


@pytest.mark.parametrize('code',[401,402])
def test_account_stops_other_roles_and_restart_without_call(tmp_path,code):
    first=AccountCircuit('test','endpoint','secret-value',tmp_path)
    a=Fake(first,code)
    with pytest.raises(LLMResponseError):a.generate_text([])
    b=Fake(AccountCircuit('test','endpoint','secret-value',tmp_path),None)
    with pytest.raises(LLMResponseError):b.generate_text([])
    assert a.calls==1 and b.calls==0
    assert 'secret-value' not in first.path.read_text()
    old=b.request_guard.status()['revision']
    with pytest.raises(ValueError):b.request_guard.resume(old-1)
    b.request_guard.resume(old)
    assert b.generate_text([])=='ok' and b.calls==1


def test_unrelated_accounts_transient_failures_and_late_response(tmp_path):
    a=AccountCircuit('test','endpoint','one',tmp_path);ticket=a.before_call()
    a.failed(LLMResponseError('bad',402),ticket)
    b=AccountCircuit('test','endpoint','two',tmp_path);assert not b.status()['blocked']
    a.resume(1);a.failed(LLMResponseError('late',402),ticket)
    assert not a.status()['blocked']
    a.failed(LLMResponseError('temporary',429),a.before_call());assert not a.status()['blocked']


def test_broken_state_is_not_implicitly_reset(tmp_path):
    a=AccountCircuit('test','endpoint','key',tmp_path);a.path.write_text('{}')
    with pytest.raises(LLMConfigError):a.before_call()


def test_vision_provider_alias_and_v1_suffix_share_account(tmp_path):
    a=AccountCircuit('qwen','https://example.test/compatible-mode/v1','key',tmp_path)
    b=AccountCircuit('dashscope','https://example.test/compatible-mode','key',tmp_path)
    assert a.scope==b.scope
    a.failed(LLMResponseError('balance',402),a.before_call())
    with pytest.raises(LLMResponseError):b.before_call()
