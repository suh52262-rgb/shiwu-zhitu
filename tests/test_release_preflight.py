import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'deployment/preflight.py'
spec = importlib.util.spec_from_file_location('release_preflight', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def prepare(root):
    (root / 'models/bge-small-zh-v1.5').mkdir(parents=True)
    (root / 'models/bge-small-zh-v1.5/config.json').write_text('{}')
    (root / 'models/bge-small-zh-v1.5/model.safetensors').write_bytes(b'test')
    (root / 'data/problems/demo').mkdir(parents=True)
    (root / 'data/problems/demo/problem.json').write_text('{}')
    (root / '.env').write_text('DEEPSEEK_API_KEY=private-never-print\n')


def test_missing_config_has_action_and_never_claims_online_success(tmp_path):
    result = module.check(tmp_path)
    assert result['status'] == 'failed'
    assert result['model_requests'] == 0
    assert next(i for i in result['checks'] if i['check'] == 'configuration')['action']


def test_configured_key_not_returned(tmp_path):
    prepare(tmp_path)
    result = module.check(tmp_path, online=True)
    assert 'private-never-print' not in str(result)
    assert next(i for i in result['checks'] if i['check'] == 'online_key_configured')['status'] == 'ready'


def test_no_key_warning_offline_but_blocks_online(tmp_path):
    prepare(tmp_path)
    (tmp_path / '.env').write_text('DEEPSEEK_API_KEY=\n')
    for online, expected in [(False, 'warning'), (True, 'failed')]:
        result = module.check(tmp_path, online=online)
        assert next(i for i in result['checks'] if i['check'] == 'online_key_configured')['status'] == expected


def test_cached_repo_name_uses_only_existing_config_and_weights(tmp_path):
    config=tmp_path/'config.json';config.write_text('{}')
    weight=tmp_path/'model.safetensors';weight.write_bytes(b'fixture')
    lookup=lambda repo,name: str(config) if name=='config.json' else str(weight) if name=='model.safetensors' else None
    assert module.model_assets_available(tmp_path,'BAAI/example',cache_lookup=lookup)
    weight.unlink()
    assert not module.model_assets_available(tmp_path,'BAAI/example',cache_lookup=lookup)
