"""Read-only local checks. Never calls a model, prints keys or repairs data."""
import argparse
import importlib.metadata
import json
import platform
import socket
import sys
from pathlib import Path


def model_assets_available(root, configured, cache_lookup=None):
    model = Path(configured)
    if not model.is_absolute():
        model = Path(root) / model
    if (model/'config.json').is_file() and (any(model.glob('*.safetensors')) or (model/'pytorch_model.bin').is_file()):
        return True
    if Path(configured).is_absolute():
        return False
    try:
        if cache_lookup is None:
            from huggingface_hub import try_to_load_from_cache
            cache_lookup = try_to_load_from_cache
        config = cache_lookup(configured, 'config.json')
        weights = [cache_lookup(configured, name) for name in ['model.safetensors', 'pytorch_model.bin']]
        return isinstance(config,str) and Path(config).is_file() and any(isinstance(p,str) and Path(p).is_file() for p in weights)
    except (ImportError, ValueError, OSError):
        return False


def check(root, *, services=False, online=False):
    root = Path(root).resolve()
    items = []

    def add(name, ok, help_text, required=True):
        items.append({'check': name, 'status': 'ready' if ok else ('failed' if required else 'warning'),
                      'action': '' if ok else help_text})

    add('python', sys.version_info[:2] == (3, 11), 'Use Python 3.11 for the locked Windows release.')
    add('platform', platform.system() == 'Windows', 'Linux/macOS are not release-validated.', False)
    for package in ['fastapi', 'streamlit', 'sqlalchemy', 'psycopg', 'qdrant-client', 'sentence-transformers']:
        try:
            importlib.metadata.version(package)
            found = True
        except importlib.metadata.PackageNotFoundError:
            found = False
        add('dependency:' + package, found, 'Install requirements-deploy.lock.txt in the project environment.')
    try:
        from dotenv import dotenv_values
        config = dotenv_values(root / '.env')
    except ImportError:
        config = {}
    add('configuration', (root / '.env').is_file(), 'Run Setup.ps1 or copy the template and configure locally.')
    model = config.get('EMBEDDING_MODEL') or 'models/bge-small-zh-v1.5'
    add('embedding_model', model_assets_available(root,model),
        'Provide the BGE model and update EMBEDDING_MODEL after moving directories.')
    add('problem_bank', any((root / 'data/problems').glob('*/problem.json')),
        'Restore the distributed demo problem bank.')
    if (root / 'manifest.json').is_file():
        sys.path.insert(0, str(root / 'scripts'))
        try:
            from deployment_bundle import verify
            verify(root)
            valid = True
        except Exception:
            valid = False
        add('manifest', valid, 'Package files differ; use a fresh extraction. Do not overwrite local work.')
    provider = config.get('LLM_PROVIDER', 'deepseek')
    key_name = {'deepseek': 'DEEPSEEK_API_KEY', 'qwen': 'QWEN_API_KEY',
                'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY'}.get(provider, 'LLM_API_KEY')
    add('online_key_configured', bool(config.get(key_name)),
        'Set your own key locally for online teaching; presence does not prove balance or model availability.', online)
    if services:
        from urllib.parse import urlparse
        try:
            q = urlparse(config.get('QDRANT_URL') or 'http://127.0.0.1:6333')
            targets = [('postgres', config.get('POSTGRES_HOST') or '127.0.0.1', int(config.get('POSTGRES_PORT') or 5432)),
                       ('qdrant', q.hostname, q.port or 6333)]
            for name, host, port in targets:
                try:
                    with socket.create_connection((host, port), timeout=3):
                        reachable = True
                except OSError:
                    reachable = False
                add(name + '_tcp', reachable, 'Start the isolated data services; TCP availability is not restored-data acceptance.')
        except ValueError:
            add('service_configuration', False, 'Check database/vector service ports in .env.')
    return {'status': 'failed' if any(i['status'] == 'failed' for i in items) else 'ready_with_warnings' if any(i['status'] == 'warning' for i in items) else 'ready',
            'checks': items, 'model_requests': 0, 'scope': 'local prerequisites only; run Accept.ps1 for restored data, then manual online acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--services', action='store_true')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = check(args.root, services=args.services, online=args.online)
    value = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(value + '\n', encoding='utf-8')
    print(value)
    raise SystemExit(1 if result['status'] == 'failed' else 0)
