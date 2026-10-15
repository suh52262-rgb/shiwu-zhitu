"""Independent regression inputs cannot become private-data or quality evidence."""
import json
import shutil
from pathlib import Path

import pytest
from tests.fixture_assets import FIXTURE_ROOT, PROBLEMS, EVALUATION, T4_DATASET, install_bank, verify_fixture_bundle


def test_fixture_manifest_checks_every_source_file_and_declares_synthetic_origin():
    assert verify_fixture_bundle() >= 20
    d = json.loads((FIXTURE_ROOT / 'manifest.json').read_text(encoding='utf8'))
    assert d['origin'] == 'ai_authored_synthetic_test_fixture'
    assert d['no_private_media'] and d['legacy_ids_are_interfaces_only']
    retrieval = json.loads((EVALUATION / 'datasets/retrieval_golden_cases.json').read_text(encoding='utf8'))
    assert len(retrieval['cases']) == 300
    assert all(c['gold']['doc_id'] == 'synthetic-unit-corpus-not-a-textbook' for c in retrieval['cases'])
    assert json.loads(T4_DATASET.read_text(encoding='utf8'))['fixture_only'] is True


def test_fixture_corruption_or_added_file_is_never_hidden(tmp_path):
    target = tmp_path / 'fixture'
    shutil.copytree(FIXTURE_ROOT, target)
    assert verify_fixture_bundle(target)
    extra = target / 'unexpected.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='file set mismatch'):
        verify_fixture_bundle(target)
    extra.unlink()
    (target / 'problems/demo_02/problem.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_fixture_bundle(target)


def test_bank_is_copy_owned_by_test_and_has_a_generated_asset(tmp_path):
    bank = install_bank(tmp_path / 'bank')
    before = (PROBLEMS / 'phy_001/problem.json').read_bytes()
    (bank / 'phy_001/problem.json').write_text('test-only change')
    assert (PROBLEMS / 'phy_001/problem.json').read_bytes() == before
    assert (bank / 'phy_002/figures/phy_002_fig1.png').read_bytes().startswith(b'\x89PNG')
    from app.services.problem_repository import ProblemRepository
    assert ProblemRepository().root != Path(__file__).resolve().parents[1] / 'data/problems'
