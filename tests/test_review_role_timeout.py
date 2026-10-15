from app.config import get_settings
from app.services.llm.factory import get_llm


def test_long_review_timeout_does_not_extend_teaching_and_preserves_model_fallback(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, 'llm_provider', 'deepseek')
    monkeypatch.setattr(settings, 'deepseek_api_key', 'placeholder-offline-only')
    monkeypatch.setattr(settings, 'diagnosis_model', 'diagnosis-override')
    monkeypatch.setattr(settings, 'review_model', '')
    monkeypatch.setattr(settings, 'llm_timeout', 30)
    monkeypatch.setattr(settings, 'review_timeout', 120)
    monkeypatch.setattr(settings, 'review_max_retries', 0)
    teaching = get_llm('diagnosis')
    review = get_llm('review')
    assert teaching.timeout == 30 and review.timeout == 120
    assert review.model == teaching.model == 'diagnosis-override'
    assert review.max_retries == 0


def test_pinned_review_uses_same_role_and_new_timeout_in_cache_identity(monkeypatch):
    from app.services import video_review_versions
    settings = get_settings()
    monkeypatch.setattr(settings, 'llm_provider', 'deepseek')
    monkeypatch.setattr(settings, 'deepseek_api_key', 'placeholder-offline-only')
    monkeypatch.setattr(settings, 'review_model', 'review-specific')
    monkeypatch.setattr(settings, 'review_timeout', 123)
    config, identity, _ = video_review_versions.generation_context()
    assert config['model'] == 'review-specific' and config['timeout'] == 123
    monkeypatch.setattr(settings, 'review_timeout', 124)
    changed, new_identity, _ = video_review_versions.generation_context()
    assert changed['timeout'] == 124 and new_identity != identity
