import hashlib
import json
from copy import deepcopy
import pytest
from app.rag.reviewed_headers import load_reviewed_headers,apply_reviewed_headers
from app.rag.section_regions import mark_running_headers,subsection_region


P={'source':'book','page':1,'body_page':3,'text':'1.6圆周运动','bbox':[10,10,100,20],'body_bbox':[10,200,200,220]}


def fixture(tmp_path):
    def write(name,data):(tmp_path/name).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    write('proposals.json',[P]);digest=hashlib.sha256((tmp_path/'proposals.json').read_bytes()).hexdigest()
    write('protocol.json',{'pdfs':{'book':{'sha256':'original'}}})
    write('visual_check.json',{'proposal_sha256':digest,'results':[{'source':'book','page':1,'body_page':3,
        'appears_running_header':True,'matching_body_heading_visible':True}]})
    return write


def test_review_hash_and_original_source_bind_exact_heading_only(tmp_path):
    fixture(tmp_path);sidecars=load_reviewed_headers(tmp_path,{'book':'original'})
    hs=[{'source':'book','page':1,'text':'1.6 圆周运动','bbox':P['bbox']},
        {'source':'book','page':1,'text':P['text'],'bbox':P['body_bbox']}]
    before=deepcopy(hs);out,applied=apply_reviewed_headers(hs,sidecars)
    assert hs==before and len(applied)==1 and out[0]['running_header_verified']
    assert not out[1].get('running_header_verified')


@pytest.mark.parametrize('change',['proposal_changed','review_missing','review_negative','duplicate_review','pdf_changed'])
def test_stale_or_incomplete_visual_reviews_are_rejected(tmp_path,change):
    write=fixture(tmp_path)
    review=json.loads((tmp_path/'visual_check.json').read_text(encoding='utf-8'))
    if change=='proposal_changed':write('proposals.json',[{**P,'text':'changed'}])
    elif change=='review_missing':review['results']=[];write('visual_check.json',review)
    elif change=='review_negative':review['results'][0]['appears_running_header']=False;write('visual_check.json',review)
    elif change=='duplicate_review':review['results']*=2;write('visual_check.json',review)
    with pytest.raises(ValueError):load_reviewed_headers(tmp_path,{'book':'changed' if change=='pdf_changed' else 'original'})


@pytest.mark.parametrize('change',[{'source':'other'},{'page':2},{'text':'1.6圆周运劫'},{'bbox':[11,10,100,20]}])
def test_other_source_page_text_or_geometry_cannot_borrow_header_witness(change):
    sidecars=[dict(P,review_proposal_sha256='fixed')]
    h={'source':'book','page':1,'text':P['text'],'bbox':P['bbox'],**change}
    out,applied=apply_reviewed_headers([h],sidecars)
    assert not applied and not out[0].get('running_header_verified')


def test_single_chapter_margin_cannot_close_real_cross_page_body():
    hs=[{'source':'book','page':1,'text':'1.6圆周运动','bbox':[20,300,180,320],'original_verified':True},
        {'source':'book','page':2,'text':'第1章质点运动学','bbox':[100,10,250,25],'original_verified':True}]
    marked=mark_running_headers(hs,{('book',1):[0,0,600,1000],('book',2):[0,0,600,1000]})
    result=subsection_region({'source':'book','page':1,'label_bbox':[400,500,450,520]},marked,inspected_pages=[1,2])
    assert result['status']=='unknown' and result['reason']=='unresolved_margin_boundary'
