import importlib.util
from pathlib import Path
import hashlib
import pytest
import json
import subprocess

spec=importlib.util.spec_from_file_location('source_bundle',Path(__file__).resolve().parents[1]/'deployment/source_bundle.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)


def test_export_blocks_unreviewed_pattern_without_leaking_it():
    token=b'sk-'+b'xyz123'*5
    with pytest.raises(ValueError) as error:
        source.validate_entries({'scripts/example.py':b'key='+token},[],{})
    assert token.decode() not in str(error.value)


def test_review_is_exact_path_rule_and_fingerprint():
    url=b'postgresql://u:fixture@127.0.0.1/'
    finding=source.scan_bytes(url)['findings'][0]
    rules={'reviewed_examples':[{'path':'tests/fixture.py',**finding}]}
    assert len(source.validate_entries({'tests/fixture.py':url},[],rules))==1
    with pytest.raises(ValueError):source.validate_entries({'backend/real.py':url},[],rules)
    with pytest.raises(ValueError):source.validate_entries({'tests/fixture.py':url.replace(b'fixture',b'changed')},[],rules)


def test_known_local_secret_is_never_waived():
    value=b'private-credential-fixture'
    rules={'reviewed_examples':[{'path':'scripts/x.py','rule':'known_local_credential','value_fingerprint':hashlib.sha256(value).hexdigest()[:16]}]}
    with pytest.raises(ValueError):source.validate_entries({'scripts/x.py':value},[value],rules)


def test_source_export_rejects_data_and_binary():
    with pytest.raises(ValueError):source.validate_entries({'data/video_runs/x/image.png':b'abc'},[],{})
    with pytest.raises(ValueError):source.validate_entries({'data/problems/private/problem.json':b'{}'},[],{})
    with pytest.raises(ValueError):source.validate_entries({'frontend/asset.bin':b'\0binary'},[],{})
    image = b'\x89PNG\r\n\x1a\n\0synthetic-reviewed-unit-asset'
    name = 'docs/assets/readme/portal.png'
    rules = {'reviewed_binary_assets': [{'path': name, 'content_type': 'image/png',
              'sha256': hashlib.sha256(image).hexdigest()}]}
    assert source.validate_entries({name: image}, [], rules) == []
    jpeg = b'\xff\xd8\xff\0synthetic-reviewed-jpeg'
    jpeg_rules = {'reviewed_binary_assets': [{'path': name, 'content_type': 'image/jpeg',
                   'sha256': hashlib.sha256(jpeg).hexdigest()}]}
    assert source.validate_entries({name: jpeg}, [], jpeg_rules) == []
    with pytest.raises(ValueError):source.validate_entries({name: image+b'changed'}, [], rules)
    with pytest.raises(ValueError):source.validate_entries({'frontend/portal.png': image}, [], rules)
    with pytest.raises(ValueError):source.validate_entries({name: image}, [b'synthetic-reviewed'], rules)



def test_only_explicit_public_label_config_is_allowed():
    valid = b'{"schema_version":1,"documents":{"test":{"title":"Test book"}}}'
    assert source.validate_entries({'data/document_names.json': valid}, [], {}) == []
    bad = b'{"schema_version":1,"documents":{"test":{"title":"Test","text":"private text"}}}'
    with pytest.raises(ValueError, match='Only display labels'):
        source.validate_entries({'data/document_names.json': bad}, [], {})
    with pytest.raises(ValueError):
        source.validate_entries({'data/other.json': valid}, [], {})


def test_export_uses_exact_commit_not_dirty_or_untracked_files(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    git('init', '-q')
    initial = {
        'backend/example.py': 'value = "committed"\n',
        'backend/__init__.py': '',
        '.gitattributes': '/third_party/** -text\n',
        'scripts/_measure_stmt_quote.py': '# internal measurement probe\n',
        'scripts/_run_miscon_eval.py': '# internal measurement probe\n',
        'examples/public_demo/README.md': 'AI authored\n',
        'requirements.txt': 'pytest\n', 'pytest.ini': '[pytest]\n',
        '.gitignore': 'artifacts/\n', 'THIRD_PARTY_NOTICES.md': 'Pending\n',
        'deployment/.env.deploy.example': 'DEEPSEEK_API_KEY=\n',
        'deployment/compose.yml': 'services: {}\n',
        'deployment/GitHub源码快速开始.md': 'Source quick start\n',
        'deployment/publication_review_rules.json': '{"reviewed_examples": []}',
        'deployment/freezes/internal.json': '{}',
    }
    for name, text in initial.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf8')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
    ref = git('rev-parse', 'HEAD')
    (root / 'backend/example.py').write_text('private dirty work')
    (root / 'backend/untracked.py').write_text('not frozen')
    lock = tmp_path / 'lock.txt'
    lock.write_text('pytest==8.4.1\n')
    clean = tmp_path / 'clean'
    source.build(root, tmp_path / 'source.zip', lock, source_ref=ref, directory=clean)
    manifest = json.loads((clean / 'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    assert manifest['source_commit'] == ref
    assert (clean / 'backend/example.py').read_text() == initial['backend/example.py']
    assert not (clean / 'backend/untracked.py').exists()
    assert not (clean / '.git').exists()
    assert not (clean / 'deployment/freezes').exists()
    assert (clean / 'backend/__init__.py').exists()
    assert (clean / '.gitattributes').read_text() == initial['.gitattributes']
    assert not (clean / 'scripts/_measure_stmt_quote.py').exists()
    assert not (clean / 'scripts/_run_miscon_eval.py').exists()
    assert (clean / 'examples/public_demo/README.md').exists()
    for name, record in manifest['files'].items():
        assert hashlib.sha256((clean / name).read_bytes()).hexdigest() == record['sha256']
    with pytest.raises(ValueError, match='clean source directory'):
        source.build(root, tmp_path / 'other.zip', lock, source_ref=ref, directory=clean)
    # New source packages retain the reviewed public front page and its vector asset.
    public_readme = '# Public project front page\n'
    (root / 'deployment/README.public.md').write_text(public_readme, encoding='utf8')
    asset = root / 'docs/assets/readme/hero.svg'
    asset.parent.mkdir(parents=True)
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding='utf8')
    git('add', 'deployment/README.public.md', 'docs/assets/readme/hero.svg')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'public docs')
    second = tmp_path / 'public-clean'
    source.build(root, tmp_path / 'public.zip', lock, directory=second)
    assert (second / 'README.md').read_text(encoding='utf8') == public_readme
    assert (second / 'docs/assets/readme/hero.svg').read_bytes() == asset.read_bytes()
