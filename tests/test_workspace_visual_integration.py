"""Read-only catalogue and workspace interaction regressions."""
import importlib
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def module(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / 'frontend'))
    return importlib.import_module(name)


def test_navigation_keeps_local_targets_even_with_invalid_port(monkeypatch):
    style = module(monkeypatch, 'workspace_style')
    monkeypatch.setenv('WORKSPACE_TEACHER_PORT', '" onclick="alert(1)')
    html = style.navigation_markup('teacher')
    assert 'http://127.0.0.1:8502' in html
    assert 'onclick' not in html and html.count('aria-current="page"') == 1
    assert html.count('target="_blank"') == 4 and 'rel="noopener noreferrer"' in html


def test_primary_and_secondary_text_contrast(monkeypatch):
    tokens = module(monkeypatch, 'workspace_style').TOKENS
    def luminance(value):
        channels = [int(value[i:i+2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in channels]
        return sum(c*w for c, w in zip(linear, [.2126,.7152,.0722]))
    for text, background in [('ink','background'),('muted','background'),('muted','surface'),('surface','primary')]:
        a, b = sorted([luminance(tokens[text]), luminance(tokens[background])])
        assert (b+.05)/(a+.05) >= 4.5


def test_textbook_catalogue_counts_recorded_pages_without_loading_or_editing_text(monkeypatch):
    data = module(monkeypatch, 'library_data')
    from app.db.tables import Base, DocumentBlock
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([DocumentBlock(id=str(i),doc_id='physics',doc_type='textbook',page=page,block_index=i,type='text',text='Original') for i,page in enumerate([1,1,3])])
        session.commit()
    assert data.catalogue(engine) == [{'doc_id':'physics','doc_type':'textbook','pages':2,'blocks':3}]
    with Session(engine) as session:
        assert session.get(DocumentBlock,'0').text == 'Original'
        session.add(DocumentBlock(id='mixed',doc_id='physics',doc_type='notes',page=3,block_index=4,type='text',text='Another type'))
        session.commit()
    assert data.catalogue(engine) == [{'doc_id':'physics','doc_type':'混合类型（需核对）','pages':2,'blocks':4}]


@pytest.mark.parametrize('failed', [False, True])
def test_material_page_separates_empty_catalogue_from_read_failure(monkeypatch, failed):
    data = module(monkeypatch, 'library_data')
    def read():
        if failed:
            raise RuntimeError('database unavailable')
        return []
    monkeypatch.setattr(data, 'catalogue', read)
    app = AppTest.from_file(str(ROOT/'frontend/material_import_app.py')).run()
    assert not app.exception
    if failed:
        assert any('读取失败' in e.value for e in app.error)
        assert len(app.metric) == 0  # failure is never represented as zero documents
    else:
        assert app.metric[0].value == '0'
    assert not app.get('file_uploader')


def test_published_question_browser_only_reads_and_does_not_display_reference_answer(monkeypatch):
    calls = []
    def request(method, url, **kwargs):
        calls.append(method)
        assert method == 'GET'
        if url.endswith('/problem-imports'):
            result = {'items':[]}
        elif url.endswith('/problems'):
            result = {'items':[{'problem_id':'demo','title':'Test physics'}]}
        else:
            result = {'title':'Test physics','statement':'Full original problem','answer':'PRIVATE REFERENCE','knowledge_points':['force']}
        return httpx.Response(200,json=result,request=httpx.Request(method,url))
    monkeypatch.setattr(httpx, 'request', request)
    app = AppTest.from_file(str(ROOT/'frontend/problem_import_app.py')).run()
    app.radio(key='authoring_workspace').set_value('已发布题库').run()
    assert not app.exception
    text = ' '.join(x.value for x in app.markdown)
    assert 'Full original problem' in text and 'PRIVATE REFERENCE' not in text
    assert all(m == 'GET' for m in calls)
