from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aggregate_rag_material_review_supplement import combine


def fixtures():
    packets = {}
    records = {}
    for reviewer, mapping in [('A', {'X': 'baseline', 'Y': 'selected'}), ('B', {'X': 'selected', 'Y': 'baseline'})]:
        packets[reviewer] = {'mapping': mapping, 'packet': {arm: [{'id': arm + '1', 'text': 'source', 'source': 'book', 'page': 1, 'type': 'text'}] for arm in ['X', 'Y']}}
        output = {arm: {'points': [{'status': 'covered' if name == 'selected' else 'partial', 'evidence_ids': [arm + '1'], 'explanation': 'evidence'}] * 2} for arm, name in mapping.items()}
        output['ocr_issues'] = []
        records[reviewer] = {'reviewer': reviewer, 'status': 'valid', 'mapping': mapping, 'output': output}
    protocol = {'input_sha256': 'same', 'dataset_sha256': 'same-data', 'rule': 'same-rule', 'packets': packets}
    original = deepcopy(records)
    original['A'] = {'reviewer': 'A', 'status': 'failed'}
    return protocol, original, {'A': records['A']}


def test_failed_only_supplement_preserves_valid_other_reviewer():
    protocol, original, supplemental = fixtures()
    result = combine(protocol, deepcopy(protocol), original, supplemental)
    assert result['valid_reviews'] == 2
    assert result['jointly_confirmed_single_case_improvement']
    assert original['A']['status'] == 'failed'


@pytest.mark.parametrize('field', ['input_sha256', 'dataset_sha256', 'rule', 'packets'])
def test_changed_material_rule_or_packet_cannot_be_combined(field):
    protocol, original, supplemental = fixtures()
    changed = deepcopy(protocol)
    changed[field] = 'different'
    with pytest.raises(ValueError):
        combine(protocol, changed, original, supplemental)


def test_valid_rating_cannot_be_replaced_with_favorable_supplement():
    protocol, original, supplemental = fixtures()
    original['A'] = deepcopy(supplemental['A'])
    with pytest.raises(ValueError):
        combine(protocol, protocol, original, supplemental)
