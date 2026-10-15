"""T4 protocol checks: denominator, freeze drift, retained failures, request accounting."""
import importlib.util
import json
from pathlib import Path

from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t4_runner", ROOT / "scripts/run_t4_batch.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_dataset_has_36_cases_and_problem_isolated_108_run_plan():
    data = runner.load_dataset(T4_DATASET)
    all_items = runner.plan(data, "all", 7)
    dev = runner.plan(data, "development", 7)
    holdout = runner.plan(data, "holdout", 7)
    assert len(all_items) == 108 and len(dev) == 72 and len(holdout) == 36
    assert {i['problem_id'] for i in dev}.isdisjoint({i['problem_id'] for i in holdout})
    assert all_items == runner.plan(data, "all", 7)
    assert len({(i['case_id'], i['repeat']) for i in all_items}) == 108


def test_synthetic_protocol_fixture_cannot_trigger_real_paid_runs(tmp_path):
    data = runner.load_dataset(T4_DATASET)
    assert data['fixture_only'] is True
    with pytest.raises(ValueError, match='cannot be used for real model evaluation'):
        runner.real_run(data, [], tmp_path, 'fixture')


def test_cli_refuses_fixture_before_claims_or_output_creation(tmp_path, monkeypatch):
    import sys
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'dataset': str(T4_DATASET), 'seed': 1}), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'verify', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['run_t4_batch', '--manifest', str(manifest),
                                    '--out', str(tmp_path / 'output'), '--split', 'holdout'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'artifacts').exists()


def test_failures_are_retained_without_automatic_case_retry(tmp_path):
    items = [{'case_id': 'c', 'repeat': i, 'status': 'planned'} for i in range(1, 4)]
    called = []
    def fail(item):
        called.append(item['repeat'])
        raise TimeoutError('result unknown')
    result = runner.execute_cases(items, fail, tmp_path, 'frozen')
    assert called == [1, 2, 3]
    assert all(r['status'] == 'failed' for r in result)
    assert len(list((tmp_path / 'runs').glob('*.json'))) == 3
    with pytest.raises(ValueError, match='already exists'):
        runner.execute_cases(items, fail, tmp_path, 'frozen')


def test_dry_records_never_become_successful_results(tmp_path):
    result = runner.execute_cases([{'case_id': 'c', 'repeat': 1}],
        lambda item: {'status': 'planned', 'executed': False}, tmp_path, 'h')
    assert result[0]['executed'] is False and result[0]['status'] == 'planned'


def test_freeze_drift_is_rejected_before_execution(tmp_path, monkeypatch):
    path = tmp_path / 'input'
    path.write_text('original')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'model_configuration', lambda: {'model':'x'})
    manifest = {'files': {'input':runner.digest(path)}, 'review_files':{}, 'configuration':{'model':'x'}}
    runner.verify(manifest)
    path.write_text('changed')
    with pytest.raises(ValueError, match='drift'):
        runner.verify(manifest)


def test_physical_failed_requests_are_counted_and_hooks_restored(monkeypatch):
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    from app.services.llm.base import LLMProvider
    from app.services.llm.schemas import TokenUsage
    original_retry = LLMProvider._with_retry
    attempts = []
    def complete(self, messages):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError('network timeout')
        return 'raw', TokenUsage(total_tokens=9)
    monkeypatch.setattr(OpenAICompatibleProvider, '_complete', complete)
    # Inject a deterministic retry boundary, preserving the real capture mechanism.
    def retry(self, messages):
        try:
            self._complete(messages)
        except TimeoutError:
            return self._complete(messages)
    monkeypatch.setattr(LLMProvider, '_with_retry', retry)
    provider = object.__new__(OpenAICompatibleProvider)
    provider.model = 'fixture'
    with runner.capture_requests() as calls:
        provider._with_retry([{'role':'user','content':'input'}])
    assert [c['network_attempt'] for c in calls] == [1, 2]
    assert calls[0]['tokens'] is None and calls[1]['tokens']['total_tokens'] == 9
    assert calls[0]['retry_group'] == calls[1]['retry_group']
    assert OpenAICompatibleProvider._complete is complete
    assert LLMProvider._with_retry is retry


def test_freeze_requires_two_reviews_of_exact_dataset(tmp_path):
    dataset = T4_DATASET
    reviews = []
    for i in range(2):
        path = tmp_path / f'r{i}.json'
        path.write_text(json.dumps({'reviewer_id':str(i),'dataset_sha256':'wrong',
            'provenance':'independent_ai_review','verdict':'accept'}))
        reviews.append(path)
    with pytest.raises(ValueError, match='this exact dataset'):
        runner.freeze(dataset, tmp_path / 'manifest.json', reviews)
    assert not (tmp_path / 'manifest.json').exists()


@pytest.mark.parametrize('summary', [
    {'dry_run':True,'smoke':False}, {'dry_run':False,'smoke':True},
    {'dry_run':False,'smoke':False},
])
def test_holdout_gate_rejects_dry_smoke_and_partial_development(tmp_path, monkeypatch, summary):
    dataset = runner.load_dataset(T4_DATASET)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    batch = tmp_path / 'batch'
    batch.mkdir()
    manifest = tmp_path / 'manifest.json'
    runner.write(manifest, {'frozen':True})
    runner.write(batch / 'summary.json', {**summary, 'evidence_sha256': runner.evidence_digest(batch)})
    gate = tmp_path / 'gate.json'
    runner.write(gate, {'passed':True, 'manifest_sha256':runner.digest(manifest),
        'batch_dir':'batch', 'summary_sha256':runner.digest(batch / 'summary.json'),
        'independent_ai_reviewers':['a','b']})
    with pytest.raises(ValueError, match='72 actual'):
        runner.check_development_gate(gate, manifest, dataset)


def test_review_packets_hide_condition_and_keep_failures(tmp_path):
    data = runner.load_dataset(T4_DATASET)
    item = runner.plan(data, 'development', 7)[0]
    record = {**item, 'status':'failed', 'model_requests':[{'model':'secret-model-name'}]}
    runner.review_packets([record], data, tmp_path, 7)
    packet = json.loads((tmp_path / 'review_packets/Q001.json').read_text(encoding='utf-8'))
    assert packet['workflow_status'] == 'failed'
    assert not any(k in packet for k in ('case_id','repeat','split','category','model_requests'))
    assert 'secret-model-name' not in json.dumps(packet)


def test_structure_repair_and_network_retry_are_distinct(monkeypatch):
    from pydantic import BaseModel
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    from app.services.llm.schemas import TokenUsage
    class Output(BaseModel):
        value: int
    outputs = iter(['bad json', '{"value": 1}'])
    def complete(self, messages):
        return next(outputs), TokenUsage(total_tokens=3)
    monkeypatch.setattr(OpenAICompatibleProvider, '_complete', complete)
    provider = OpenAICompatibleProvider(api_key='fixture', base_url='http://fixture.invalid',
        model='fixture', client=object())
    with runner.capture_requests() as calls:
        assert provider.generate_structured([{'role':'user','content':'input'}], Output).value == 1
    assert [c['network_attempt'] for c in calls] == [1, 1]
    assert [c['structured_pass'] for c in calls] == [1, 2]
    assert calls[0]['structured_group'] == calls[1]['structured_group']


def test_full_gate_binds_locked_reviews_to_actual_outputs(tmp_path, monkeypatch):
    data = runner.load_dataset(T4_DATASET)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    batch = tmp_path / 'batch'
    manifest = tmp_path / 'manifest.json'
    runner.write(manifest, {'frozen':True})
    for item in runner.plan(data, 'development', 1):
        runner.write(batch / 'runs' / f"{item['case_id']}-{item['repeat']}.json", {
            **item, 'manifest_sha256':runner.digest(manifest), 'status':'succeeded', 'output':'original fixture'})
    summary = batch / 'summary.json'
    runner.write(summary, {'dry_run':False, 'smoke':False, 'evidence_sha256':runner.evidence_digest(batch)})
    reviews = []
    for i in range(2):
        path = tmp_path / f'review{i}.json'
        runner.write(path, {'reviewer_id':str(i),'manifest_sha256':runner.digest(manifest),
            'summary_sha256':runner.digest(summary),'evidence_sha256':runner.evidence_digest(batch),
            'provenance':'independent_ai_review','verdict':'accept','initial_ratings_locked':True,
            'blind_item_ids':[f'Q{n:03d}' for n in range(1,73)],
            'metrics':{'severe_false_accusations':0,'unsupported_cognitive_claims':0,
                'invalid_displayed_references':0,'serious_physics_contradictions':0,
                'explicit_error_total':12,'explicit_error_recognized':10}})
        reviews.append({'path':path.name,'sha256':runner.digest(path)})
    gate = tmp_path / 'gate.json'
    runner.write(gate, {'passed':True, 'manifest_sha256':runner.digest(manifest),
        'batch_dir':'batch', 'summary_sha256':runner.digest(summary),'review_reports':reviews})
    runner.check_development_gate(gate, manifest, data)
    record = next((batch / 'runs').glob('*.json'))
    payload = json.loads(record.read_text(encoding='utf-8'))
    runner.write(record, {**payload,'output':'changed physics, same counts'})
    with pytest.raises(ValueError, match='outputs/review packet drift'):
        runner.check_development_gate(gate, manifest, data)
