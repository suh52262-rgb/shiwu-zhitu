"""Exact preservation of locked ratings and first-use holdout protection."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import normalize_t4_ratings as normalizer
import run_t4_batch as runner


def fixture(tmp_path):
    batch = tmp_path / 'batch'
    batch.mkdir()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    (batch / 'summary.json').write_text('{}', encoding='utf-8')
    item = {'ratable': True, 'correctness': 4, 'targeting': 3, 'actionability': 2,
            'answer_leakage': 1, 'material_has_definite_error': False,
            'explicit_error_detected': False, 'severe_false_accusation': False,
            'unsupported_cognitive_claim': False, 'invalid_displayed_reference': False,
            'serious_physics_contradiction': False,
            'evidence': {'quote': '原文 α=−2；不得改写', 'nested': [False, 1, None]}}
    raw = {'reviewer_id': 'fixture', 'provenance': 'independent_ai_review',
           'initial_ratings_locked': True, 'manifest_sha256': normalizer.digest(manifest),
           'summary_sha256': normalizer.digest(batch / 'summary.json'),
           'evidence_sha256': normalizer.evidence_digest(batch),
           'initial_issue_counts': {'custom': 7}, 'metrics': {'custom': 8},
           'ratings': [{**copy.deepcopy(item), 'blind_item_id': f'Q{i:03d}'} for i in range(1, 73)]}
    source = tmp_path / 'initial.json'
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    return source, tmp_path / 'normalized.json', batch, manifest, raw


def test_file_adapter_preserves_all_values_and_original(tmp_path):
    source, out, batch, manifest, raw = fixture(tmp_path)
    original = source.read_bytes()
    result = normalizer.normalize(source, out, batch, manifest)
    assert source.read_bytes() == original
    normalizer.verify_round_trip(raw, json.loads(out.read_text(encoding='utf-8')),
                                 result['format_adapter']['mapping'])
    assert result['metrics'] == raw['metrics']
    assert result['initial_issue_counts'] == raw['initial_issue_counts']
    assert result['format_adapter']['source_sha256'] == normalizer.digest(source)
    with pytest.raises(ValueError, match='Output exists'):
        normalizer.normalize(source, out, batch, manifest)
    assert source.read_bytes() == original


@pytest.mark.parametrize('field,value', [('correctness_1to5', 5),
    ('severe_false_accusation', True), ('evidence', 'changed'), ('ratable', 1)])
def test_round_trip_detects_value_and_type_changes(tmp_path, field, value):
    *_, raw = fixture(tmp_path)
    adapted, mapping = normalizer.adapt(raw)
    adapted['items'][0][field] = value
    with pytest.raises(ValueError, match='Round-trip changed'):
        normalizer.verify_round_trip(raw, adapted, mapping)


@pytest.mark.parametrize('mutation', ['dual_lists', 'dual_scores', 'unlocked', 'missing_id', 'bad_hash'])
def test_reject_ambiguous_or_unbound_inputs_without_output(tmp_path, mutation):
    source, out, batch, manifest, raw = fixture(tmp_path)
    if mutation == 'dual_lists': raw['items'] = raw['ratings']
    if mutation == 'dual_scores': raw['ratings'][0]['correctness_1to5'] = 4
    if mutation == 'unlocked': raw['initial_ratings_locked'] = False
    if mutation == 'missing_id': raw['ratings'].pop()
    if mutation == 'bad_hash': raw['summary_sha256'] = 'wrong'
    source.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError):
        normalizer.normalize(source, out, batch, manifest)
    assert not out.exists()


def test_canonical_schema_and_duplicate_json_keys(tmp_path):
    source, out, batch, manifest, raw = fixture(tmp_path)
    canonical, _ = normalizer.adapt(raw)
    source.write_text(json.dumps(canonical), encoding='utf-8')
    normalizer.normalize(source, out, batch, manifest)
    source.write_text('{"reviewer_id":"a","reviewer_id":"b"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        normalizer.normalize(source, tmp_path / 'second.json', batch, manifest)


@pytest.mark.parametrize('infrastructure_crash', [False, True])
def test_first_holdout_claims_precede_execution_and_survive_failure(tmp_path, monkeypatch,
                                                                  infrastructure_crash):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path / 'manifest.json'
    identities = ['a' * 64, 'b' * 64]
    runner.write(manifest, {'dataset': 'fixture.json', 'seed': 1,
                            'holdout_question_identities': identities})
    monkeypatch.setattr(runner, 'verify', lambda _: None)
    monkeypatch.setattr(runner, 'load_dataset', lambda _: {})
    monkeypatch.setattr(runner, 'plan', lambda *args: [])
    monkeypatch.setattr(runner, 'check_development_gate', lambda *args: None)
    monkeypatch.setattr(runner, 'review_packets', lambda *args: None)
    claims = tmp_path / 'artifacts/t4_holdout_claims'
    assert not claims.exists()
    calls = []
    def execute(*args):
        calls.append({p.stem: json.loads(p.read_text())['manifest_sha256']
                      for p in claims.glob('*.json')})
        if infrastructure_crash:
            raise RuntimeError('fixture infrastructure failure')
        return []
    monkeypatch.setattr(runner, 'real_run', execute)
    def invoke(out):
        monkeypatch.setattr(sys, 'argv', ['run_t4_batch', '--manifest', str(manifest),
            '--out', str(tmp_path / out), '--split', 'holdout', '--development-gate', 'fixture'])
        runner.main()
    invoke('first')
    assert len(list(claims.glob('*.json'))) == 2
    with pytest.raises(SystemExit):
        invoke('second')
    assert calls == [{identity: runner.digest(manifest) for identity in identities}]
    assert not (tmp_path / 'second').exists()


def test_rejected_gate_does_not_reserve_or_execute_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    manifest = tmp_path / 'manifest.json'
    runner.write(manifest, {'dataset': 'fixture.json', 'seed': 1,
                            'holdout_question_identities': ['a' * 64]})
    monkeypatch.setattr(runner, 'verify', lambda _: None)
    monkeypatch.setattr(runner, 'load_dataset', lambda _: {})
    monkeypatch.setattr(runner, 'plan', lambda *args: [])
    def reject(*args): raise ValueError('fixture rejected gate')
    monkeypatch.setattr(runner, 'check_development_gate', reject)
    monkeypatch.setattr(runner, 'real_run', lambda *args: pytest.fail('Must not execute'))
    monkeypatch.setattr(sys, 'argv', ['run_t4_batch', '--manifest', str(manifest),
        '--out', str(tmp_path / 'out'), '--split', 'holdout', '--development-gate', 'fixture'])
    with pytest.raises(ValueError, match='rejected gate'):
        runner.main()
    assert not list((tmp_path / 'artifacts/t4_holdout_claims').glob('*.json'))
    assert not (tmp_path / 'out').exists()
