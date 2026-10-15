from copy import deepcopy
import pytest
from app.rag.formula_positions import verify_printed_label, label_position_links

ANCHOR = {'reference': '3.17', 'label_text': '(3.17)', 'label_bbox': [80, 20, 120, 40]}
SPAN = {'text': '（3.17）', 'bbox': [82, 22, 118, 38]}
PAGE = [0, 0, 600, 800]


def test_full_label_survives_nearby_prose_without_substring_matching():
    spans = [SPAN, {'text': '文字', 'bbox': [79, 19, 110, 21]}]
    before = deepcopy((ANCHOR, spans))
    result = verify_printed_label(ANCHOR, spans, PAGE)
    assert result['status'] == 'verified'
    assert result['original_label_text'] == SPAN['text']
    assert not result['formula_bbox_known'] and not result['semantic_support_validated']
    assert (ANCHOR, spans) == before


@pytest.mark.parametrize('spans', [
    [{'text': '由(3.17)得', 'bbox': SPAN['bbox']}],
    [SPAN, {**SPAN, 'bbox': [200, 20, 240, 40]}],
    [{'text': '(3.', 'bbox': [82, 22, 100, 38]},
     {'text': '17)', 'bbox': [100, 22, 118, 38]}],
    [{**SPAN, 'text': '(3.18)'}], [],
])
def test_prose_duplicate_split_and_wrong_labels_remain_unknown(spans):
    assert verify_printed_label(ANCHOR, spans, PAGE)['status'] == 'rejected'


@pytest.mark.parametrize('box', [[200, 20, 240, 40], [0, 0, 0, 2],
                               [0, float('nan'), 120, 40], [0, 0, 700, 900]])
def test_original_span_requires_valid_page_and_anchor_geometry(box):
    assert verify_printed_label(ANCHOR, [{**SPAN, 'bbox': box}], PAGE)['status'] == 'rejected'


def test_anchor_text_cannot_disagree_with_reference():
    assert verify_printed_label({**ANCHOR, 'label_text': '(3.18)'}, [SPAN], PAGE)['status'] == 'rejected'


def test_invalid_anchor_is_rejected():
    assert verify_printed_label({**ANCHOR, 'label_bbox': [-3, 0, 120, 40]}, [SPAN], PAGE)['status'] == 'rejected'


@pytest.mark.parametrize('target', ['formula', 'text'])
def test_sqlite_integer_false_does_not_bypass_label_filter(target):
    facts = [{'id': 'f', 'doc_id': 'book', 'page': 1, 'type': 'formula',
              'text': r'x=y\tag{3.17}', 'indexable': 1},
             {'id': 't', 'doc_id': 'book', 'page': 1, 'type': 'text',
              'text': '(3.17)', 'bbox': ANCHOR['label_bbox'], 'indexable': 1}]
    next(b for b in facts if b['type'] == target)['indexable'] = 0
    assert not label_position_links(facts)[0]
