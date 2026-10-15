"""Filtering never silently selects a new attempt; aliases retain source identity."""
import importlib
import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def frontend(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / 'frontend'))
    return importlib.import_module(name)


def test_tag_matching_any_all_and_query_do_not_read_reference_answers(monkeypatch):
    ui = frontend(monkeypatch, 'problem_browser')
    problems = {
        'a': {'title':'题 A', 'knowledge_points':['动量','碰撞'], 'answer':'secret'},
        'b': {'title':'题 B', 'knowledge_points':['碰撞']},
        'c': {'title':'题 C', 'knowledge_points':[]},
    }
    assert ui.matching_ids(problems, tags=['动量','碰撞']) == ['a','b']
    assert ui.matching_ids(problems, tags=['动量','碰撞'], require_all=True) == ['a']
    assert ui.matching_ids(problems, 'Ａ 动量') == ['a']
    assert ui.matching_ids(problems, 'secret') == []
    assert ui.matching_ids(problems, 'c') == ['c']  # untagged remains searchable


def test_missing_tags_are_not_inferred_or_parsed_from_a_string(monkeypatch):
    ui = frontend(monkeypatch, 'problem_browser')
    assert ui.problem_tags({'knowledge_points':'动量'}) == []
    assert ui.problem_tags({'knowledge_points':['动量',' ',None,'动量']}) == ['动量']
    assert ui.selection_options([], 'current') == ['current']
    assert ui.selection_options(['other'], 'current') == ['current','other']


def install_tagged_api(monkeypatch):
    from tests.test_student_problem_statement_ui import install_api
    install_api(monkeypatch)
    original = httpx.get
    def get(url, **kwargs):
        response = original(url, **kwargs)
        if url.endswith('/problems'):
            payload = response.json()
            payload['items'][0]['knowledge_points'] = ['动量','单位换算']
            payload['items'][1]['knowledge_points'] = ['机械能','单位换算']
            return httpx.Response(200,json=payload,request=httpx.Request('GET',url))
        return response
    monkeypatch.setattr(httpx,'get',get)
    monkeypatch.setattr(httpx,'post',lambda *a,**k: (_ for _ in ()).throw(AssertionError('filtering must not submit')))


def test_filters_preserve_current_problem_draft_and_timeline_until_explicit_selection(monkeypatch):
    install_tagged_api(monkeypatch)
    app = AppTest.from_file(str(ROOT/'frontend/streamlit_app.py')).run()
    app.text_area(key='answer_input').set_value('p1 unsent').run()
    app.session_state['timeline']=[{'type':'submit','content':'old step','timestamp':1}]
    app.multiselect(key='problem_tags').set_value(['机械能']).run()
    assert not app.exception
    assert app.selectbox(key='problem_select').value == 'p1'
    assert app.text_area(key='answer_input').value == 'p1 unsent'
    assert app.session_state['timeline'][0]['content']=='old step'
    assert len(app.selectbox(key='problem_select').options)==2
    app.selectbox(key='problem_select').set_value('p2').run()
    assert app.selectbox(key='problem_select').value=='p2'
    app.text_area(key='answer_input').set_value('p2 unsent').run()
    app.text_input(key='problem_query').set_value('no such problem').run()
    assert app.selectbox(key='problem_select').value=='p2'
    assert app.selectbox(key='problem_select').disabled
    assert app.text_area(key='answer_input').value=='p2 unsent'
    assert any('没有匹配题目' in i.value for i in app.info)
    app.button(key='clear_problem_filters').click().run()
    assert not app.exception
    assert app.selectbox(key='problem_select').value=='p2'
    assert app.text_area(key='answer_input').value=='p2 unsent'
    app.selectbox(key='problem_select').set_value('p1').run()
    assert app.text_area(key='answer_input').value=='p1 unsent'


def test_all_tags_mode_really_filters_the_choices_without_switching(monkeypatch):
    install_tagged_api(monkeypatch)
    app = AppTest.from_file(str(ROOT/'frontend/streamlit_app.py')).run()
    app.multiselect(key='problem_tags').set_value(['动量','单位换算']).run()
    assert len(app.selectbox(key='problem_select').options)==2
    app.checkbox(key='problem_tags_all').check().run()
    assert not app.exception
    assert app.selectbox(key='problem_select').value=='p1'
    assert len(app.selectbox(key='problem_select').options)==1


def test_book_names_and_search_retain_doc_ids_and_unknown_books(monkeypatch):
    labels = frontend(monkeypatch,'resource_labels')
    assert labels.document_title('jiaocai-5-dianci')=='新概念物理教程·电磁学'
    assert labels.document_matches('jiaocai-5-dianci','电磁学')
    assert labels.document_matches('jiaocai-5-dianci','JIAOCAI-5')
    assert labels.document_matches('jiaocai-1-shangce','大学物理学 上册')
    assert not labels.document_matches('jiaocai-1-shangce','下册')
    assert labels.document_title('my-new-book')=='my-new-book'
    assert 'jiaocai-5-dianci' in labels.document_label('jiaocai-5-dianci')


def test_material_search_displays_book_name_but_preview_uses_original_identity(monkeypatch):
    data = frontend(monkeypatch,'library_data')
    calls=[]
    monkeypatch.setattr(data,'catalogue',lambda:[{'doc_id':'jiaocai-5-dianci','doc_type':'textbook','pages':1,'blocks':1},
                                               {'doc_id':'jiaocai-6-guangxue','doc_type':'textbook','pages':1,'blocks':1}])
    monkeypatch.setattr(data,'material_preview',lambda doc_id: calls.append(doc_id) or {'blocks':[],'runs':[]})
    app=AppTest.from_file(str(ROOT/'frontend/material_import_app.py')).run()
    next(t for t in app.text_input if t.label=='按教材名称或编号查找').set_value('电磁学').run()
    assert not app.exception
    assert app.selectbox[0].value=='jiaocai-5-dianci'
    assert app.selectbox[0].options==['新概念物理教程·电磁学 · jiaocai-5-dianci']
    assert calls[-1]=='jiaocai-5-dianci'
    assert app.dataframe[0].value.iloc[0]['教材名称']=='新概念物理教程·电磁学'


@pytest.mark.parametrize('content', ['not-json','[]','{"schema_version":9,"documents":{}}'])
def test_bad_display_registry_falls_back_without_inventing_names(monkeypatch,tmp_path,content):
    labels=frontend(monkeypatch,'resource_labels')
    registry=tmp_path/'names.json'
    registry.write_text(content,encoding='utf-8')
    labels.document_names.cache_clear()
    monkeypatch.setattr(labels,'REGISTRY',registry)
    try:
        assert labels.document_title('unknown')=='unknown'
    finally:
        labels.document_names.cache_clear()
