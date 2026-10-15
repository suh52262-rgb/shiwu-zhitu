from copy import deepcopy
from app.rag.section_regions import mark_running_headers,subsection_region


def heading(text,page,y=10,source='book'):
    return {'source':source,'page':page,'text':text,'bbox':[10,y,180,y+15],'original_verified':True}


def bounds():return {('book',n):[0,0,600,1000] for n in [1,2,3]}


def test_repeated_real_subsections_in_margin_are_never_removed():
    hs=[heading('4.1方程的推导',1),heading('4.1方程的推导',2),heading('4.2方程应用',3,100)]
    before=deepcopy(hs);marked=mark_running_headers(hs,bounds())
    assert hs==before and all(not h.get('running_header_verified') for h in marked)
    result=subsection_region({'source':'book','page':2,'label_bbox':[400,150,450,170]},marked,inspected_pages=[1,2,3])
    assert result['reason']=='repeated_heading'


def test_parent_running_headers_require_distinct_original_pages():
    hs=[heading('第五章连续体力学',1),heading('第五章连续体力学',2)]
    assert all(h['running_header_verified'] for h in mark_running_headers(hs,bounds()))
    assert not any(h.get('running_header_verified') for h in mark_running_headers([hs[0],hs[0]],bounds()))


def test_opaque_ocr_title_needs_exact_body_witness_not_guessed_section_symbol():
    hs=[heading('84.伯努利方程及其应用',1),heading('84.伯努利方程及其应用',2)]
    assert not any(h.get('running_header_verified') for h in mark_running_headers(hs,bounds()))
    with_body=hs+[heading(hs[0]['text'],1,500)]
    marked=mark_running_headers(with_body,bounds())
    assert all(h.get('running_header_verified') for h in marked[:2])
    assert not marked[2].get('running_header_verified') and marked[0]['text']==hs[0]['text']


def test_header_is_not_borrowed_from_another_book_or_invalid_page_box():
    hs=[heading('第五章连续体力学',1),heading('第五章连续体力学',2,source='other')]
    assert not any(h.get('running_header_verified') for h in mark_running_headers(hs,bounds()))
    assert not any(h.get('running_header_verified') for h in mark_running_headers(
        [heading('第五章连续体力学',1),heading('第五章连续体力学',2)],{}))


def test_two_columns_with_overlapping_heading_height_are_not_ordered_by_list():
    hs=[heading('4.1方程推导',1,100),{**heading('4.2方程应用',1,100),'bbox':[350,100,550,115]},heading('4.3其他内容',1,400)]
    for order in [hs,list(reversed(hs))]:
        result=subsection_region({'source':'book','page':1,'label_bbox':[400,200,450,220]},order,inspected_pages=[1])
        assert result['reason']=='overlapping_heading_order'
