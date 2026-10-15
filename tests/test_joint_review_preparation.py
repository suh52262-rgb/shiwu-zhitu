import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_joint_delivery_review import validate_material

POOL = [dict(id='a', source='book', page=1, text='第一句。第二句。')]
BASE = dict(sources=[dict(id='a', source='book', page=1, text='第一句。')])


def test_failed_baseline_fragment_is_valid_without_masquerading_as_full_candidate():
    validate_material(POOL, BASE, dict(selection_status='failed', sources=BASE['sources']))
    with pytest.raises(ValueError):
        validate_material(POOL, BASE, dict(selection_status='ready', sources=BASE['sources']))


def test_foreign_source_changed_fallback_and_budget_are_rejected():
    for result in [dict(selection_status='ready', sources=[{**POOL[0], 'source': 'other'}]),
                   dict(selection_status='failed', sources=POOL),
                   dict(selection_status='ready', sources=[{**POOL[0], 'text': '编写的原文'}])]:
        with pytest.raises(ValueError):
            validate_material(POOL, BASE, result)
    with pytest.raises(ValueError):
        validate_material(POOL, BASE, dict(selection_status='ready', sources=POOL), budget=1)
