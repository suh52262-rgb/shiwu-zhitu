import pytest
from app.rag.reference_candidates import document_reference_candidates

A = {'id': 'anchor', 'source': 'book', 'page': 10, 'type': 'text', 'text': '由式（1.2）得到'}
F = {'id': 'formula', 'doc_id': 'book', 'page': 1, 'type': 'formula', 'text': r'W=Q^2/2C\tag{1.2}', 'indexable': True}

def test_unique_distant_reference_preserves_fact_and_anchor():
    linked, omissions = document_reference_candidates([A], [F])
    assert linked[0]['text'] == F['text'] and linked[0]['page'] == 1
    assert linked[0]['context_link']['anchor_ids'] == ['anchor']
    assert not omissions

def test_cross_book_or_ambiguous_reference_is_not_resolved():
    assert not document_reference_candidates([A], [{**F, 'doc_id': 'other'}])[0]
    assert not document_reference_candidates([A], [F, {**F, 'id': 'duplicate'}])[0]

def test_filtered_generated_and_empty_formulas_remain_excluded():
    for change in [{'indexable': False}, {'is_generated': True}, {'text': '', 'formula_number': '1.2'}]:
        linked, omissions = document_reference_candidates([A], [{**F, **change}])
        assert not linked and omissions[0]['reason'] == 'filtered'

def test_existing_ids_limit_and_absent_reference_are_not_invented():
    assert document_reference_candidates([A, {**F, 'source': 'book'}], [F]) == ([], [])
    assert not document_reference_candidates([{**A, 'text': '没有编号的解释'}], [F])[0]
    assert document_reference_candidates([A], [F], limit=0)[1][0]['reason'] == 'limit'
    with pytest.raises(ValueError):
        document_reference_candidates([A], [F], limit=-1)
