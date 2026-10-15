"""Public effective configuration. Never expose keys, endpoints or local paths."""
import hashlib
import os

from app.agent.prompt_registry import PROMPT_ROLES, get_prompt, resolve_version
from app.config import get_settings
from app.services.llm.factory import _PROVIDERS, _resolve_model
from app.services.rag_materials import policy


def _safe_options(options):
    roles = {'step_matcher', 'analyze_step', 'diagnosis', 'rerank', 'tutor', 'review', 'graph_generator', 'graph_validator'}
    return {role: {key: value for key, value in values.items()
                   if (key == 'thinking' and value in ('enabled', 'disabled'))
                   or (key == 'max_tokens' and type(value) is int and value > 0)}
            for role, values in options.items() if role in roles and isinstance(values, dict)}


def public_runtime_configuration():
    settings = get_settings()
    spec = _PROVIDERS.get(settings.llm_provider)
    roles = ('step_matcher', 'analyze_step', 'diagnosis', 'tutor', 'review', 'graph_generator', 'graph_validator')
    return {
        'profile': os.environ.get('SHIWU_RUNTIME_PROFILE', 'custom'),
        'provider': settings.llm_provider,
        'llm_role_options': _safe_options(settings.llm_role_options),
        'photo_role_options': _safe_options(settings.photo_role_options) if settings.llm_provider == 'deepseek' else {},
        'photo_analysis_budget_s': settings.photo_analysis_budget_s,
        'embedding_warmup_enabled': settings.embedding_warmup_enabled,
        'models': {role: _resolve_model(settings, role, spec.default_model if spec else '') for role in roles},
        'prompts': {role: {'version': resolve_version(role),
                          'sha256': hashlib.sha256(get_prompt(role).encode('utf-8')).hexdigest()}
                    for role in PROMPT_ROLES},
        'rag_material_policy': policy(),
        'timeouts_s': {'teaching': settings.llm_timeout, 'review': settings.review_timeout},
        'network_retries': {'teaching':settings.llm_max_retries, 'review':settings.review_max_retries},
        'acceptance_status': 'current_source_requires_revalidation',
        'historical_acceptance': 'v18: deepseek-flash / review v3 / graphless text / empty corpus',
    }
