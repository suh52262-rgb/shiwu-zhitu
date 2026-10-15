from types import SimpleNamespace

from app.rag import pdf_import


def test_merged_paragraph_box_contains_indented_and_unindented_lines(monkeypatch):
    lines = [dict(text='第一行', bbox=[96, 10, 400, 20]),
             dict(text='第二行', bbox=[74, 27, 420, 37]),
             dict(text='第三行', bbox=[74, 44, 410, 54])]
    monkeypatch.setattr(pdf_import, 'text_lines', lambda page: lines)
    result = pdf_import._paragraphs(SimpleNamespace())
    assert result == [dict(text='第一行第二行第三行', bbox=[74, 10, 420, 54])]
    for line in lines:
        b, s = result[0]['bbox'], line['bbox']
        assert b[0] <= s[0] and b[1] <= s[1] and b[2] >= s[2] and b[3] >= s[3]


def test_out_of_order_and_overlapping_lines_are_not_silently_merged(monkeypatch):
    for second in [[74, 5, 420, 15], [74, 15, 420, 25]]:
        monkeypatch.setattr(pdf_import, 'text_lines', lambda page: [
            dict(text='甲', bbox=[74, 10, 400, 20]), dict(text='乙', bbox=second)])
        assert len(pdf_import._paragraphs(SimpleNamespace())) == 2


def test_gap_and_column_boundaries_preserve_original_text_order(monkeypatch):
    monkeypatch.setattr(pdf_import, 'text_lines', lambda page: [
        dict(text='甲', bbox=[74, 10, 400, 20]),
        dict(text='乙', bbox=[74, 40, 400, 50]),
        dict(text='丙', bbox=[450, 52, 570, 62])])
    result = pdf_import._paragraphs(SimpleNamespace())
    assert [b['text'] for b in result] == ['甲', '乙', '丙']
