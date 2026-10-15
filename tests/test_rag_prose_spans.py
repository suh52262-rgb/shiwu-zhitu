from copy import deepcopy

import pytest

from app.rag.prose_spans import prose_companions

A = dict(id='a', doc_id='book', page=1, type='text', text='这一公式是在特定条件下建立的', bbox=[90, 10, 400, 20], indexable=True)
B = dict(id='b', doc_id='book', page=1, type='text', text='，并不能直接用于任意过程。', bbox=[70, 28, 350, 38], indexable=True)
P = dict(id='a', source='book', page=1, type='text', text=A['text'], truncated=False)


def run(facts=None, verified=None, **kwargs):
    return prose_companions([P], facts or [A, B], verified if verified is not None else ['a', 'b'], **kwargs)


def test_complete_source_is_added_without_claiming_semantic_link_or_paragraph_start():
    before = deepcopy([A, B]); out, details = run()
    assert [s['text'] for s in out] == [A['text'], B['text']]
    assert [A, B] == before and out[0] == P
    assert details['spans'][0]['end_complete'] is True
    assert details['spans'][0]['start_complete'] == 'unknown'
    assert details['spans'][0]['semantic_support_validated'] is False


@pytest.mark.parametrize('change', [dict(page=2), dict(doc_id='other'), dict(indexable=0),
    dict(is_generated=True), dict(type='figure_caption'), dict(bbox=[]),
    dict(bbox=[70, 60, 350, 70]), dict(bbox=[70, 15, 350, 25]),
    dict(bbox=[120, 28, 400, 38]), dict(bbox=[410, 28, 550, 38])])
def test_page_filter_geometry_and_indentation_barriers_do_not_join(change):
    assert len(run([A, {**B, **change}])[0]) == 1


def test_unverified_interposed_formula_caption_duplicate_and_sentence_end_stop():
    assert len(run(verified=['a'])[0]) == 1
    barrier = dict(id='x', doc_id='book', page=1, type='text', text='x=y', bbox=[200, 21, 250, 26], indexable=False)
    assert len(run([A, barrier, B])[0]) == 1
    assert len(run([A, B, B])[0]) == 1
    assert len(prose_companions([{**P, 'text': A['text']+'。'}], [{**A, 'text': A['text']+'。'}, B], ['a', 'b'])[0]) == 1


def test_whole_span_budget_and_source_prefix_conflict():
    assert run(budget=len(B['text'])-1)[0] == [P]
    out, details = prose_companions([{**P, 'text': '不同材料'}], [A, B], ['a', 'b'])
    assert len(out) == 1 and details['omissions'][0]['reason'] == 'source_prefix_mismatch'
    with pytest.raises(ValueError):
        run(budget=-1)


def test_truncated_anchor_is_restored_only_from_verified_same_source():
    out, details = prose_companions([{**P, 'text': A['text'][:3], 'truncated': True}], [A], ['a'])
    assert out[0]['text'] == A['text'] and not out[0]['truncated']
    assert details['extra_characters'] == len(A['text'])-3
