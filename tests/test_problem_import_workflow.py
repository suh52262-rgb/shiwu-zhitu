import io
import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.services.problem_import import ProblemImporter, ImportConflict, safe_id
from app.services.problem_repository import ProblemRepository
from app.rag.exam_split import split_pages


def png():
    out=io.BytesIO()
    Image.new('RGB',(300,200),'white').save(out,format='PNG')
    return out.getvalue()


@pytest.fixture
def importer(tmp_path):
    return ProblemImporter(tmp_path/'drafts',tmp_path/'problems')


def create_reviewed(importer,pid='new_question',crop=True):
    record=importer.create(png(),'question.png',use_vision=True,
        reader=lambda _:('1. A body accelerates at 2 m/s^2. Find its velocity after 3 s.\n【参考答案】\nv=6 m/s',{'model':'test-double'}))
    item=record['items'][0]
    item=importer.edit(record['import_id'],item['item_id'],base_revision=0,
        values={'problem_id':pid,'title':'Acceleration','crops':[{'page':1,'bbox':[10,10,100,80]}] if crop else []},
        reviewer='test operator',confirmed=True)
    return record,item


def test_complete_publication_preserves_source_revision_and_registered_figure(importer):
    record,item=create_reviewed(importer)
    assert not importer.problems_root.exists()
    publication=importer.publish(record['import_id'],item['item_id'],1)
    repo=ProblemRepository(importer.problems_root)
    question=repo.get_problem(publication['problem_id'])
    assert 'Find its velocity' in question.statement and question.answer=='v=6 m/s'
    assert question.figures==['figures/figure_1.png']
    with Image.open(repo.figure_path(question.problem_id,question.figures[0])) as image:
        assert image.size==(90,70)
    assert (importer.folder(record['import_id'])/'source.png').read_bytes()==png()
    assert not (repo.problem_dir(question.problem_id)/'source.png').exists()
    stored=importer.load(record['import_id'])['items'][0]
    assert stored['original_answer']=='v=6 m/s' and len(stored['history'])==1
    assert importer.publish(record['import_id'],item['item_id'],1)==publication


def test_draft_and_stale_revision_cannot_publish(importer):
    record=importer.create(png(),'image.png')
    item=importer.add(record['import_id'])
    with pytest.raises(ImportConflict):importer.publish(record['import_id'],item['item_id'],0)
    importer.edit(record['import_id'],item['item_id'],base_revision=0,values={'source_pages':[1],'statement':'complete statement'})
    with pytest.raises(ImportConflict):importer.edit(record['import_id'],item['item_id'],base_revision=0,values={})
    with pytest.raises(ImportConflict):importer.publish(record['import_id'],item['item_id'],0)


def test_edit_invalidates_review_and_preserves_original(importer):
    record,item=create_reviewed(importer)
    edited=importer.edit(record['import_id'],item['item_id'],base_revision=1,values={'answer':'different'})
    assert edited['review'] is None and edited['original_answer']=='v=6 m/s'
    with pytest.raises(ImportConflict):importer.publish(record['import_id'],item['item_id'],2)


def test_duplicate_id_keeps_old_problem_and_its_graph(importer):
    record,item=create_reviewed(importer)
    importer.publish(record['import_id'],item['item_id'],1)
    graph=importer.problems_root/'new_question'/'graph.json'
    graph.write_text('old graph',encoding='utf-8')
    before=(graph.parent/'problem.json').read_bytes()
    other,other_item=create_reviewed(importer)
    with pytest.raises(ImportConflict):importer.publish(other['import_id'],other_item['item_id'],1)
    assert (graph.parent/'problem.json').read_bytes()==before and graph.read_text()=='old graph'


def test_recover_published_directory_after_manifest_write_interrupt(importer,monkeypatch):
    from app.services import problem_import
    record,item=create_reviewed(importer)
    real=problem_import.atomic_json
    def fail_manifest(path,data):
        if path.name=='manifest.json':raise OSError('simulated interruption')
        real(path,data)
    monkeypatch.setattr(problem_import,'atomic_json',fail_manifest)
    with pytest.raises(OSError):importer.publish(record['import_id'],item['item_id'],1)
    monkeypatch.setattr(problem_import,'atomic_json',real)
    assert importer.publish(record['import_id'],item['item_id'],1)['problem_id']=='new_question'


@pytest.mark.parametrize('pid',['../x','x/y','x\\y','C:bad','CON','x ',''])
def test_path_and_reserved_names_rejected(pid):
    with pytest.raises(ValueError):safe_id(pid)


def test_bad_crop_and_unconfirmed_placeholders_rejected(importer):
    record,item=create_reviewed(importer)
    with pytest.raises(ValueError):importer.edit(record['import_id'],item['item_id'],base_revision=1,
        values={'crops':[{'page':1,'bbox':[0,0,900,40]}]})
    with pytest.raises(ValueError):importer.edit(record['import_id'],item['item_id'],base_revision=1,
        values={'statement':'[待核对]'},reviewer='operator',confirmed=True)
    assert importer.load(record['import_id'])['items'][0]['revision']==1


def test_scan_without_vision_explicitly_needs_transcription(importer):
    doc=pymupdf.open()
    page=doc.new_page()
    page.insert_image(page.rect,stream=png())
    record=importer.create(doc.tobytes(),'scan.pdf')
    assert record['status']=='draft' and record['pages'][0]['status']=='needs_transcription'
    assert record['items']==[] and not importer.problems_root.exists()


def test_vision_failure_does_not_silently_claim_no_questions(importer):
    def fail(_):raise RuntimeError('network failed')
    record=importer.create(png(),'scan.png',use_vision=True,reader=fail)
    assert record['pages'][0]['status']=='failed'
    assert 'network failed' in record['pages'][0]['vision_error']
    assert record['items']==[]


def test_corrupt_file_is_recorded_as_failed_not_success(importer):
    record=importer.create(b'not a PDF','bad.pdf')
    assert record['status']=='failed' and record['error']
    assert importer.load(record['import_id'])['status']=='failed'


def test_cross_page_options_and_answer_empty_marker_are_preserved():
    drafts=split_pages([(1,'1. Find the correct answer to this long question.\nA. first option\nB. second option\n解'),(2,'solution continued\n2. The next long question is on the next page.')])
    assert 'A. first option\nB. second option' in drafts[0].statement
    assert drafts[0].solution=='solution continued'
    assert drafts[0].pages==[1,2]

@pytest.mark.parametrize('marker',['解析：','解答：','Answer:','Solution:'])
def test_explicit_solution_labels_do_not_become_student_statement(marker):
    drafts=split_pages([(1,f'1. A sufficiently long physics question.\n{marker}\nsecret solution')])
    assert drafts[0].solution=='secret solution'
    assert 'secret' not in drafts[0].statement


def test_api_publication_student_read_and_no_static_reference_leak(tmp_path,monkeypatch):
    monkeypatch.setenv('PROBLEM_IMPORTS_DIR',str(tmp_path/'drafts'))
    monkeypatch.setenv('PROBLEMS_DIR',str(tmp_path/'bank'))
    from app.main import app
    client=TestClient(app)
    created=client.post('/problem-imports',files={'file':('scan.png',png(),'image/png')}).json()
    iid=created['import_id']
    item=client.post(f'/problem-imports/{iid}/items').json()
    edited=client.put(f'/problem-imports/{iid}/items/{item["item_id"]}',json={'base_revision':0,
        'values':{'problem_id':'imported_test','statement':'Full statement with subquestions (1) and (2)',
                  'answer':'secret answer','source_pages':[1],'crops':[{'page':1,'bbox':[10,10,100,80]}]},
        'reviewer':'test operator','confirmed':True})
    assert edited.status_code==200
    published=client.post(f'/problem-imports/{iid}/items/{item["item_id"]}/publish',json={'base_revision':1})
    assert published.status_code==200
    student=client.get('/problems/imported_test')
    assert student.status_code==200 and student.json()['statement'].endswith('(1) and (2)')
    assert 'answer' not in student.json() and 'secret answer' not in student.text
    assert client.get(student.json()['figure_urls'][0]).status_code==200
    assert client.get('/problem-assets/imported_test/problem.json').status_code==404
    assert client.get('/problem-assets/imported_test/graph.json').status_code==404
    assert client.get(f'/problem-imports/{iid}/pages/1').status_code==200
    assert client.post(f'/problem-imports/{iid}/items/{item["item_id"]}/publish',json={'base_revision':0}).status_code==409
