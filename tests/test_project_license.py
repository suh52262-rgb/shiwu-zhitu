"""Keep the chosen project license and exact third-party evidence attached."""
import hashlib
import json
from pathlib import Path
import importlib.util
from email.message import Message
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_project_license_is_agpl_version_three_only_with_canonical_text():
    assert 'SPDX-License-Identifier: AGPL-3.0-only' in (ROOT / 'NOTICE').read_text(encoding='utf8')
    source = json.loads((ROOT / 'third_party/project-license-source.json').read_text(encoding='utf8'))
    text = (ROOT / 'LICENSE').read_bytes()
    assert hashlib.sha256(text).hexdigest() == source['sha256']
    assert b'GNU AFFERO GENERAL PUBLIC LICENSE' in text
    assert source['spdx'] == 'AGPL-3.0-only'


def test_exact_dependency_notices_have_verifiable_texts_and_no_missing_record():
    inventory = json.loads((ROOT / 'third_party/inventory.json').read_text(encoding='utf8'))
    assert inventory['distribution_count'] == len(inventory['distributions']) == 116
    assert inventory['gaps'] == []
    assert inventory['project_license'] == 'AGPL-3.0-only'
    for row in inventory['distributions']:
        assert row['license_files'], row['name']
        for record in row['license_files']:
            path = ROOT / 'third_party' / record['file']
            assert path.resolve().is_relative_to((ROOT / 'third_party').resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']


@pytest.mark.parametrize('has_license', [True, False])
def test_license_collector_records_hash_or_explicit_gap(monkeypatch, tmp_path, has_license):
    spec = importlib.util.spec_from_file_location('unit_license_collector', ROOT / 'scripts/collect_third_party_licenses.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = tmp_path / 'LICENSE'
    text.write_text('Synthetic package license fixture', encoding='utf8')
    md = Message()
    md['Name'] = 'unit-package'
    md['License'] = 'MIT'
    fake = SimpleNamespace(version='1.0', metadata=md,
        files=['unit-package-1.0.dist-info/LICENSE'] if has_license else [], locate_file=lambda _: text)
    monkeypatch.setattr(module.metadata, 'distribution', lambda _: fake)
    lock = tmp_path / 'lock.txt'
    lock.write_text('unit-package==1.0\n')
    output = tmp_path / 'notices'
    result = module.collect(lock, output, include_pip=False)
    assert result['gaps'] == ([] if has_license else ['unit-package'])
    if has_license:
        f = result['distributions'][0]['license_files'][0]
        assert hashlib.sha256((output / f['file']).read_bytes()).hexdigest() == f['sha256']
    fake.version = '2.0'
    with pytest.raises(ValueError, match='differs from the lock'):
        module.collect(lock, tmp_path / 'other', include_pip=False)
