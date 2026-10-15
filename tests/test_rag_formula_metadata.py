import copy

import pytest

from app.rag.formula_metadata import recover_missing_numbers, simple_equation


F = dict(id='f', doc_id='book', page=1, type='formula', text='$$dA = pdV$$')
T = dict(id='t', doc_id='book', page=1, type='text', text='(10.3)', bbox=[90, 20, 110, 30])
L = [dict(page=1, text='dA =pdV', bbox=[30, 20, 70, 30]),
     dict(page=1, text='(10.3)', bbox=[90, 20, 110, 30])]


def test_exact_source_geometry_proposal_does_not_mutate_facts():
    facts = copy.deepcopy([F, T]); before = copy.deepcopy(facts)
    result, omissions = recover_missing_numbers(facts, L)
    assert facts == before and not omissions
    assert result[0]['reference'] == '10.3'
    assert result[0]['original_formula_text'] == 'dA =pdV'
    assert result[0]['subsection_relation'] == 'unknown'
    assert result[0]['semantic_support_validated'] is False


@pytest.mark.parametrize('change', [dict(text='dA=pSdl'), dict(page=2),
                                  dict(bbox=[30, 40, 70, 50]), dict(bbox=[100, 20, 120, 30])])
def test_different_equation_page_or_geometry_cannot_recover(change):
    assert not recover_missing_numbers([F, T], [{**L[0], **change}, L[1]])[0]


@pytest.mark.parametrize('fact,change', [(F, dict(indexable=False)), (F, dict(is_generated=True)),
                                      (T, dict(indexable=False)), (T, dict(bbox=[])),
                                      (T, dict(bbox=[90, 200, 110, 210])),
                                      (F, dict(indexable=0)), (T, dict(indexable=0))])
def test_filters_and_position_disagreement_reject(fact, change):
    facts = [{**b, **change} if b is fact else b for b in [F, T]]
    assert not recover_missing_numbers(facts, L)[0]


def test_duplicate_formula_or_label_or_pdf_line_reject():
    for extra in [{**F, 'id': 'f2'}, {**T, 'id': 't2'}, {**F, 'text': '$$x=y$$'}]:
        assert not recover_missing_numbers([F, T, extra], L)[0]
    assert not recover_missing_numbers([F, T], L+[L[0]])[0]
    assert not recover_missing_numbers([F, T], L+[L[1]])[0]


def test_existing_number_is_not_rewritten_and_cross_document_input_rejects():
    assert not recover_missing_numbers([{**F, 'formula_number': '10.3'}, T], L)[0]
    with pytest.raises(ValueError):
        recover_missing_numbers([F, {**T, 'doc_id': 'other'}], L)


def test_no_latex_repair_prose_or_ocr_substitution():
    assert simple_equation('$$dA = pdV$$') == 'dA=pdV'
    for text in [r'$$A=\int pdV$$', '这里dA=pdV', 'dA＝pdV', 'x', 'x=y\\tag{1.2}']:
        assert simple_equation(text) is None
