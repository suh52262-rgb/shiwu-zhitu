from app.services.runtime_configuration import public_runtime_configuration


def test_runtime_configuration_is_safe_and_does_not_claim_current_acceptance(monkeypatch):
    from app.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, 'deepseek_api_key', 'private-test-key')
    monkeypatch.setattr(settings, 'tutor_model', 'role-specific-model')
    result = public_runtime_configuration()
    assert result['models']['tutor'] == 'role-specific-model'
    assert result['acceptance_status'] == 'current_source_requires_revalidation'
    assert 'private-test-key' not in str(result)
    assert result['prompts']['review']['sha256']

def test_role_options_never_publish_unknown_fields(monkeypatch):
    from app.config import get_settings
    settings=get_settings()
    monkeypatch.setattr(settings,'llm_role_options',{'tutor':{'api_key':'secret','thinking':'disabled','max_tokens':1200},'unknown':{'api_key':'secret'}})
    result=public_runtime_configuration()
    assert result['llm_role_options']=={'tutor':{'thinking':'disabled','max_tokens':1200}}
    assert 'secret' not in str(result)
