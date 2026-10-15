from pathlib import Path
import importlib
import json
import httpx
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def test_student_workspace_switch_keeps_unsubmitted_draft_without_post(monkeypatch):
    from tests.test_student_problem_statement_ui import install_api
    install_api(monkeypatch)
    monkeypatch.setattr(httpx,'post',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not submit')))
    app=AppTest.from_file(str(ROOT/'frontend/streamlit_app.py')).run()
    app.text_area(key='answer_input').input('m a = F - f').run()
    for page in ['学习首页','视频复盘','学习记录','学习作答']:
        app.radio(key='student_workspace').set_value(page).run()
        assert not app.exception
    assert app.text_area(key='answer_input').value=='m a = F - f'
    assert app.session_state['attempt_id'] is None


def test_drafts_are_scoped_to_problem(monkeypatch):
    from tests.test_student_problem_statement_ui import install_api
    install_api(monkeypatch)
    app=AppTest.from_file(str(ROOT/'frontend/streamlit_app.py')).run()
    app.text_area(key='answer_input').input('p1 unsent').run()
    app.selectbox(key='problem_select').set_value('p2').run()
    assert app.text_area(key='answer_input').value==''
    app.text_area(key='answer_input').input('p2 unsent').run()
    app.selectbox(key='problem_select').set_value('p1').run()
    assert app.text_area(key='answer_input').value=='p1 unsent'


def test_reference_progress_does_not_reveal_equations_and_ignores_foreign_edges(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    ui=importlib.import_module('learning_workspace')
    graph={'nodes':[{'id':'S1','equation':'SECRET FORMULA','description':'SECRET SOLUTION'},{'id':'S2'}],
           'edges':[['S1','S2'],['S2','foreign']]}
    dot=ui.progress_dot(graph,{'current_node':'S2','visited_nodes':['S1']})
    assert 'SECRET' not in dot and 'foreign' not in dot
    assert '已有匹配记录' in dot and '当前核对' in dot


def test_reference_target_rejects_previous_revision(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    ui=importlib.import_module('learning_workspace')
    rows=[{'source_id':'one','source_revision':2}]
    assert ui.target_round(rows,'one',1) is None
    assert ui.target_round(rows,'other',2) is None
    assert ui.target_round(rows,'one',2)==0


def test_review_retries_reuse_id_but_different_decisions_do_not(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    ui=importlib.import_module('teacher_workbench')
    state={};body={'attempt_id':'a','decision':'confirmed','note':'original'}
    assert ui.review_request_id(state,'s','e',body)==ui.review_request_id(state,'s','e',dict(body))
    assert ui.review_request_id(state,'s','e',{**body,'decision':'dismissed'})!=ui.review_request_id(state,'s','e',body)


def test_teacher_save_requires_operator_and_explicit_confirmation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    monkeypatch.setattr(httpx,'post',lambda *a,**k: (_ for _ in ()).throw(AssertionError('no permission to save')))
    source='''
import streamlit as st
from teacher_workbench import render_teacher_tasks
class Data:
    def attempts(self,**kwargs):return []
errors={'review_store_status':'ready','counts':{'unreviewed':1},'evidence':[{'evidence_id':'e','review_status':'unreviewed','student_id':'s','problem_id':'p','concept':'force','student_text':'F=ma','issue':'written relation','attempt_id':'a','source_revision':0}]}
render_teacher_tasks(Data(),errors,'http://localhost:8000',lambda aid:st.caption('evidence'))
'''
    app=AppTest.from_string(source).run()
    next(b for b in app.button if b.label=='保存核对决定').click().run()
    assert not app.exception
    assert any('请填写核对者' in w.value for w in app.warning)


def test_restore_link_opens_video_and_its_problem_without_post(monkeypatch):
    from tests.test_student_problem_statement_ui import install_api
    install_api(monkeypatch)
    monkeypatch.setattr(httpx,'post',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not submit')))
    app=AppTest.from_file(str(ROOT/'frontend/streamlit_app.py'))
    app.query_params.update(video_job='owned-job',video_job_problem='p2',video_job_student='stu_demo')
    app.run()
    assert not app.exception
    assert app.radio(key='student_workspace').value=='视频复盘'
    assert app.selectbox(key='problem_select').value=='p2'
    assert app.radio(key='media_mode').value=='📹 传一段视频'
    # An explicit problem change still wins over the URL default.
    app.selectbox(key='problem_select').set_value('p1').run()
    assert app.selectbox(key='problem_select').value=='p1'


def test_material_preview_is_document_scoped_and_preserves_original_flags(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.db.tables import Base,DocumentBlock,DocumentImportRun
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    data=importlib.import_module('library_data')
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([
            DocumentBlock(id='first',doc_id='book',doc_type='textbook',page=2,block_index=0,type='text',text='Exact OCR',indexable=False,is_generated=True,source_kind='vl'),
            DocumentBlock(id='later',doc_id='book',doc_type='textbook',page=4,block_index=0,type='text',text='Later'),
            DocumentBlock(id='foreign',doc_id='other',doc_type='textbook',page=1,block_index=0,type='text',text='Foreign'),
            DocumentImportRun(id='run',doc_id='book',status='partial',pages_total=3,pages_ok=1),
            DocumentImportRun(id='other-run',doc_id='other',status='ready')])
        session.commit()
    result=data.material_preview('book',engine=engine)
    assert [r['id'] for r in result['blocks']]==['first']
    assert result['blocks'][0]['text']=='Exact OCR'
    assert result['blocks'][0]['indexable'] is False and result['blocks'][0]['is_generated'] is True
    assert [r['id'] for r in result['runs']]==['run']
    assert result['runs'][0]['status']=='partial'


def test_video_browser_uses_real_round_fields_and_does_not_borrow_another_run(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    def get(url,**kwargs):
        return httpx.Response(200,json={'images':[{'run_id':'other','evidence_id':'foreign','timestamp_sec':40}]},request=httpx.Request('GET',url))
    monkeypatch.setattr(httpx,'get',get)
    source="""
from learning_workspace import render_evidence_browser
import httpx
render_evidence_browser({'run_id':'own','media_type':'video','rounds':[{'round':2,'ts':40,'answer':'Exact answer','source_id':'src','source_revision':3}]},httpx.get,'http://localhost:8000','s','p','a')
"""
    app=AppTest.from_string(source).run()
    assert not app.exception
    assert app.selectbox[0].options==['第 2 轮 · 40 秒']
    assert any('Exact answer' in c.value for c in app.code)
    assert not app.get('image')
    assert any('本次来源没有可读取' in i.value for i in app.info)


def test_evidence_failure_is_not_reported_as_no_images(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'frontend'))
    monkeypatch.setattr(httpx,'get',lambda url,**k:httpx.Response(503,request=httpx.Request('GET',url)))
    source="""
from learning_workspace import render_evidence_browser
import httpx
render_evidence_browser({'run_id':'own','rounds':[{'answer':'OCR'}]},httpx.get,'http://localhost:8000','s','p','a')
"""
    app=AppTest.from_string(source).run()
    assert not app.exception
    assert any('原图读取暂未成功' in w.value for w in app.warning)
    assert not any('没有可读取' in i.value for i in app.info)
