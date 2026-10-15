from types import SimpleNamespace
import pytest
from app.services.llm.openai_compatible import OpenAICompatibleProvider
from app.services.llm.base import LLMResponseError
from app.services.llm.factory import get_llm, clear_cache
from app.config import get_settings


def test_explicit_options_reach_request_and_length_does_not_become_success():
    calls=[]
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'),finish_reason='length')])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm=OpenAICompatibleProvider('key','https://api.deepseek.com','deepseek-flash',client=client,thinking='disabled',max_tokens=256)
    with pytest.raises(LLMResponseError):llm.chat('system','user')
    assert calls[0]['extra_body']=={'thinking':{'type':'disabled'}}
    assert calls[0]['max_tokens']==256


def test_role_options_participate_in_factory_identity(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','deepseek');monkeypatch.setenv('DEEPSEEK_API_KEY','test-key')
    monkeypatch.setenv('LLM_ROLE_OPTIONS','{"rerank":{"thinking":"disabled","max_tokens":256}}')
    get_settings.cache_clear();clear_cache()
    try:
        a=get_llm('rerank');b=get_llm('tutor')
        assert a is not b and a.thinking=='disabled' and b.thinking==''
        assert get_llm('rerank') is a
    finally:get_settings.cache_clear();clear_cache()
