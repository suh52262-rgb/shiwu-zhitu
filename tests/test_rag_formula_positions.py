import pytest
from app.rag.formula_positions import label_position_links,standalone_label
F={'id':'f','doc_id':'book','page':1,'type':'formula','text':r'x=y\tag{1.2}','indexable':True}
T={'id':'t','doc_id':'book','page':1,'type':'text','text':'（1.2）','bbox':[1,2,3,4],'indexable':True}

def test_unique_label_preserves_position_without_claiming_formula_box():
    linked,_=label_position_links([F,T]);assert linked[0]['label_bbox']==T['bbox']
    assert linked[0]['position_kind']=='printed_label_only' and not linked[0]['formula_bbox_known']

def test_prose_reference_ambiguous_and_cross_page_do_not_anchor():
    for other in [{**T,'text':'由（1.2）式可得'}, {**T,'page':2}, {**T,'doc_id':'other'}]:
        assert not label_position_links([F,other])[0]
    assert not label_position_links([F,T,{**T,'id':'other'}])[0]
    assert not label_position_links([F,T,{**F,'id':'other'}])[0]

def test_conflicting_number_filtered_and_unnumbered_are_unknown():
    for change in [{'formula_number':'1.3'},{'indexable':False},{'is_generated':True},{'text':'x=y'}]:
        assert not label_position_links([{**F,**change},T])[0]

@pytest.mark.parametrize('box',[[],[0,0,0,2],[0,float('nan'),2,3]])
def test_invalid_coordinates_do_not_become_positions(box):
    assert not label_position_links([F,{**T,'bbox':box}])[0]

def test_exact_standalone_label_and_fullwidth_punctuation():
    assert standalone_label('（3． 19）')=='3.19'
    assert standalone_label('式（3.19）') is None
    assert standalone_label('(3-19)') is None
