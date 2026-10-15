from copy import deepcopy
import pytest
from app.rag.section_regions import heading_kind,subsection_region,region_text_candidates,verify_text_geometry

A={'source':'book','page':2,'label_bbox':[300,30,350,40]}
H1={'source':'book','page':1,'bbox':[10,50,150,60],'text':'4.1 方程的推导','original_verified':True}
H2={'source':'book','page':2,'bbox':[10,90,150,100],'text':'4. 2 方程的应用','original_verified':True}
B={'id':'b','doc_id':'book','page':1,'type':'text','bbox':[10,70,400,80],
   'text':'仅在这些条件下适用，不能用于其他情况。','indexable':1}


def test_cross_page_bounded_scope_preserves_negation_and_never_claims_continuity():
    region=subsection_region(A,[H1,H2],inspected_pages=[1,2]);assert region['status']=='bounded'
    caption={**B,'id':'caption','type':'figure_caption','bbox':[10,81,90,85]}
    before=deepcopy([B,caption]);out=region_text_candidates(region,[B,caption],['b','caption'])
    assert len(out)==1 and out[0]['text']==B['text'] and [B,caption]==before
    assert not out[0]['relation']['semantic_support_validated']
    assert not out[0]['relation']['paragraph_continuity_validated']


@pytest.mark.parametrize('text',['4.1 如图所示，','4.1 x=y','4.1?','例4.1方程推导','4.l方程推导'])
def test_no_ocr_repair_or_prose_heading(text):assert heading_kind(text) is None


@pytest.mark.parametrize('headings,pages',[
    ([H1],[1,2]),([H2],[1,2]),([H1,H2],[2]),
    ([H1,{**H2,'original_verified':False}],[1,2]),
    ([H1,{**H1,'page':2,'bbox':[10,10,150,20]},H2],[1,2]),
    ([{**H1,'text':'§4.伯努利方程及其应用'},H2],[1,2]),
])
def test_missing_repeated_parent_and_unverified_ranges_remain_unknown(headings,pages):
    assert subsection_region(A,headings,inspected_pages=pages)['status']=='unknown'


def test_same_page_new_subsection_excludes_other_section():
    h={**H1,'page':2,'bbox':[10,10,150,20]}
    region=subsection_region(A,[h,H2],inspected_pages=[2])
    inside={**B,'page':2,'bbox':[10,45,400,55]}
    outside={**inside,'id':'out','bbox':[10,110,400,120]}
    assert [b['id'] for b in region_text_candidates(region,[inside,outside],['b','out'])]==['b']


@pytest.mark.parametrize('change',[{'indexable':0},{'is_generated':True},{'doc_id':'other'},
                                 {'bbox':[]},{'bbox':[10,50,400,80]}])
def test_source_filter_and_boundary_overlap_not_added(change):
    region=subsection_region(A,[H1,H2],inspected_pages=[1,2])
    assert not region_text_candidates(region,[{**B,**change}],['b'])


def test_unverified_and_duplicate_text_identity_not_added():
    region=subsection_region(A,[H1,H2],inspected_pages=[1,2])
    assert not region_text_candidates(region,[B],[])
    assert not region_text_candidates(region,[B,B],['b'])


def test_verified_running_header_does_not_replace_body_subsection():
    header={**H1,'page':2,'bbox':[10,1,150,8],'text':'第五章连续体力学','running_header_verified':True}
    assert subsection_region(A,[H1,header,H2],inspected_pages=[1,2])['status']=='bounded'


def test_unknown_heading_outside_selected_interval_does_not_poison_it():
    opaque={**H1,'bbox':[10,1,150,8],'text':'4.l方程推导','original_verified':False}
    assert subsection_region(A,[opaque,H1,H2],inspected_pages=[1,2])['status']=='bounded'
    opaque={**opaque,'page':2,'bbox':[10,10,150,20]}
    assert subsection_region(A,[H1,opaque,H2],inspected_pages=[1,2])['status']=='unknown'


def test_running_header_identity_and_non_prose_labels_are_excluded():
    region=subsection_region(A,[H1,H2],inspected_pages=[1,2])
    label={**B,'id':'label','text':'a,b'}
    assert not region_text_candidates(region,[B,label],['b','label'],excluded_ids=['b'])


def test_overflow_box_sidecar_requires_exact_bounded_text_and_no_mutation():
    b={**B,'bbox':[10,10,110,30]};before=deepcopy(b);seen=[]
    def extract(clip):seen.append(clip);return b['text']+'\n'
    witness=verify_text_geometry(b,[0,0,100,100],extract)
    assert witness['coordinate_recovery'] and witness['bbox']==[10,10,100,30]
    assert witness['stored_bbox']==before['bbox'] and b==before
    assert seen[0][2]<=100
    assert verify_text_geometry(b,[0,0,100,100],lambda box:b['text'][:-1])['status']=='rejected'


@pytest.mark.parametrize('box',[[101,0,120,20],[0,0,float('nan'),20],[0,0,0,20]])
def test_invalid_text_geometry_never_extracts_or_repairs(box):
    def forbidden(clip):raise AssertionError('must not extract')
    assert verify_text_geometry({**B,'bbox':box},[0,0,100,100],forbidden)['status']=='rejected'
