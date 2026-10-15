import pytest

from app.rag.pdf_import import _extract_formula_number


@pytest.mark.parametrize('latex,expected', [
    (r'$$Q=\Delta E+A\tag{10.2}$$', '(10.2)'),
    (r'$$dA=pdV (10.3)$$', '(10.3)'),
    (r'$$x=y\tag{3．19}$$', '(3.19)'),
    (r'$$x=y（3．19）$$', '(3.19)'),
    (r'$$x=y(3-19)$$', '(3-19)'),
    (r'$$x=y\tag{3.19}(3.19)$$', '(3.19)'),
    (r'$$x=y\tag{3.19}(3.20)$$', ''),
    (r'$$x=y\tag{3.19}\tag{3.20}$$', ''),
    (r'$$dA=pdV$$', ''),
    (r'$$x=y\tag{A}$$', ''),
])
def test_formula_number_recognizes_explicit_tag_without_inventing_missing_number(latex, expected):
    assert _extract_formula_number(latex) == expected


def test_collected_formula_embeds_explicit_tag_number(monkeypatch):
    from types import SimpleNamespace
    from app.rag import pdf_import

    monkeypatch.setattr(pdf_import, 'page_kind', lambda page: 'native_text')
    monkeypatch.setattr(pdf_import, '_paragraphs', lambda page: [])
    monkeypatch.setattr(pdf_import, 'find_captions', lambda page: [])
    monkeypatch.setattr(pdf_import, '_read_page_with_vl', lambda page: (
        {'formulas': [r'$$dA=pdV\tag{10.3}$$'], 'figures': []}, {'model': 'test'}))
    result = pdf_import.collect_page(SimpleNamespace(rect=SimpleNamespace(width=600, height=800)),
                                     299, doc_id='book')
    formula = next(b for b in result.blocks if b['type'] == 'formula')
    assert formula['formula_number'] == '(10.3)'
    assert '公式(10.3)' in formula['embed_text']
    assert formula['source_kind'] == 'vl' and formula['doc_id'] == 'book'
