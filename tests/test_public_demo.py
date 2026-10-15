"""Public examples are self-contained, numerically checked and non-destructive."""
import importlib.util
import json
from pathlib import Path

import pytest
from app.models.problem import Problem
from app.models.problem_graph import ProblemGraph
from app.services.problem_repository import ProblemRepository
from app.services.graph_validator import validate_graph

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = load('public_demo_seed', ROOT / 'scripts/prepare_public_demo.py')


def test_public_examples_validate_and_do_not_claim_human_approval(tmp_path, monkeypatch):
    result = seed.install(tmp_path)
    assert set(result['installed']) == {'public_force_01', 'public_impulse_02'}
    repo = ProblemRepository(tmp_path)
    for pid in result['installed']:
        p = repo.get_problem(pid)
        g = repo.get_graph(pid)
        assert p.figures == [] and p.metadata['uses_private_material'] is False
        assert g.metadata['needs_review'] and g.metadata['human_review_status'] == 'pending'
        report = validate_graph(g, p, use_llm=False)
        assert report.checks['structure'] == 'pass'
        monkeypatch.setenv('GRAPH_RELEASE_POLICY', 'published-only')
        assert repo.get_teaching_graph(pid) is None
        monkeypatch.setenv('GRAPH_RELEASE_POLICY', 'legacy-demo')
        assert repo.get_teaching_graph(pid) is not None
    assert seed.install(tmp_path)['installed'] == []
    assert len(seed.install(tmp_path)['unchanged']) == 2


def test_demo_numbers_are_independently_recomputed():
    from decimal import Decimal as D
    root = ROOT / 'examples/public_demo/problems'
    force = Problem.model_validate_json((root / 'public_force_01/problem.json').read_bytes())
    impulse = Problem.model_validate_json((root / 'public_impulse_02/problem.json').read_bytes())
    assert (D('10') - D('2')) / D('2') == 4
    assert (D('10') - D('2')) / D('2') * D('3') == 12
    assert D('0.5') * (D('-1') - D('3')) == -2
    assert D('0.5') * (D('-1') - D('3')) / D('0.2') == -10
    assert '12.0 m/s' in force.answer and '−10.0 N' in impulse.answer


def test_conflicting_demo_causes_no_partial_install(tmp_path):
    p = tmp_path / 'public_impulse_02'
    p.mkdir()
    (p / 'problem.json').write_text('private work', encoding='utf8')
    with pytest.raises(ValueError, match='nothing overwritten'):
        seed.install(tmp_path)
    assert not (tmp_path / 'public_force_01').exists()
    assert (p / 'problem.json').read_text() == 'private work'


def test_corrupt_source_refused_before_writing(tmp_path):
    import shutil
    source = tmp_path / 'source'
    shutil.copytree(ROOT / 'examples/public_demo', source)
    p = source / 'problems/public_force_01/problem.json'
    p.write_text('{}')
    destination = tmp_path / 'dest'
    with pytest.raises(ValueError, match='hash mismatch'):
        seed.install(destination, source)
    assert not destination.exists()


def test_media_is_synthetic_and_never_overwrites_inputs(tmp_path):
    media = load('public_demo_media', ROOT / 'examples/public_demo/make_sample_media.py')
    result = media.generate(tmp_path)
    from PIL import Image
    assert 'not real student' in result['source']
    assert len(result['files']) == 4
    assert Image.open(tmp_path / result['files'][0]).size == (960, 640)
    with pytest.raises(ValueError, match='not be overwritten'):
        media.generate(tmp_path)
