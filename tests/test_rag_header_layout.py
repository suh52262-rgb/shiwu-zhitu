from copy import deepcopy
import pytest
from app.rag.header_layout import header_layout_proposals


def facts():
    top={'source':'book','page':1,'text':'1.6圆周运动','bbox':[100,10,200,25],
        'font_size':10,'original_verified':True,'font_witness_verified':True}
    return [top,{**top,'page':2},{**top,'page':3,'bbox':[20,200,250,230],'font_size':16}]


def bounds():return {('book',n):[0,0,600,1000] for n in [1,2,3]}


def test_proposal_needs_unique_body_size_contrast_and_still_requires_visual_check():
    hs=facts();before=deepcopy(hs);p,o=header_layout_proposals(hs,bounds())
    assert len(p)==2 and not o and hs==before
    assert all(not x['running_header_verified'] and x['status']=='needs_original_visual_check' for x in p)


@pytest.mark.parametrize('variant',['no_body','duplicate_body','no_repeat','same_size','other_book','wrong_ocr','no_font_witness','invalid_font','out_of_page'])
def test_margins_and_font_proxy_alone_do_not_verify_header(variant):
    hs=facts()
    if variant=='no_body':hs=hs[:2]
    elif variant=='duplicate_body':hs.append({**hs[2],'bbox':[20,300,250,330]})
    elif variant=='no_repeat':hs=[hs[0],hs[2]]
    elif variant=='same_size':hs[2]['font_size']=10
    elif variant=='other_book':hs[2]['source']='other'
    elif variant=='wrong_ocr':hs[2]['text']='1.6圆周运劫'
    elif variant=='no_font_witness':hs[2]['font_witness_verified']=False
    elif variant=='invalid_font':hs[2]['font_size']=float('nan')
    elif variant=='out_of_page':hs[2]['bbox']=[20,200,700,230]
    assert not header_layout_proposals(hs,bounds())[0]


def test_only_explicit_contents_witness_removes_toc_duplicate():
    hs=facts();entry={**hs[2],'bbox':[20,300,250,330]}
    assert not header_layout_proposals(hs+[entry],bounds())[0]
    p,_=header_layout_proposals(hs+[{**entry,'contents_page_verified':True}],bounds())
    assert len(p)==2
