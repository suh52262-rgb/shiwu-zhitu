import pytest
from app.rag.evidence_companions import evidence_companions,equation_references
F={'id':'f','doc_id':'book','page':3,'type':'formula','text':r'x=y\tag{1.2}','indexable':True}
A={'id':'a','source':'book','doc_id':'book','page':2,'type':'text','text':'利用式（1.2）可核对该物理关系。','indexable':True}
T={'id':'t','doc_id':'book','page':3,'type':'text','text':'（1.2）式只在这个条件下适用。','indexable':True}
P={'formula_id':'f','source':'book','page':3,'reference':'1.2','label_id':'label','source_check':'verified_label_text'}

def test_only_explicit_equation_grammar_is_accepted():
    assert equation_references('式（1.2）和(1.2)式')==['1.2']
    assert not equation_references('图（1.2），表（1.2），以及（1.2）')

def test_cross_page_reference_retains_sources_without_inventing_subsection():
    added,_=evidence_companions([A],[A,F,T],[P])
    assert [b['id'] for b in added]==['f','t']
    assert added[0]['text']==F['text'] and added[1]['page']==3
    assert all(b['relation']['subsection_relation']=='unknown' for b in added)

def test_formula_anchor_can_find_explicit_explanation():
    assert {b['id'] for b in evidence_companions([{**F,'source':'book'}],[A,F,T],[P])[0]}=={'a','t'}

def test_unverified_or_wrong_label_and_ambiguous_formulas_are_rejected():
    for p in [[],[{**P,'page':4}],[{**P,'source':'other'}],[{**P,'source_check':'label_clip_mismatch'}]]:
        assert not evidence_companions([A],[A,F,T],p)[0]
    assert not evidence_companions([A],[A,F,{**F,'id':'other'}],[P])[0]

def test_filtered_and_cross_document_texts_do_not_enter():
    assert [b['id'] for b in evidence_companions([A],[A,F,{**T,'indexable':False},{**T,'id':'other','doc_id':'other'}],[P])[0]]==['f']
    assert not evidence_companions([A],[A,{**F,'is_generated':True},T],[P])[0]

def test_budget_keeps_whole_blocks_and_baseline_unchanged():
    original=dict(A);added,omitted=evidence_companions([A],[A,F,T],[P],budget=1)
    assert not added and omitted and A==original
    with pytest.raises(ValueError):evidence_companions([],[],[],limit=-1)


@pytest.mark.parametrize('field,value', [('indexable',0),('is_generated',True)])
def test_filtered_anchor_cannot_request_formula(field,value):
    bad={**A,field:value}
    added,omitted=evidence_companions([bad],[bad,F,T],[P])
    assert not added and omitted[0]['reason']=='invalid_or_filtered_anchor'


@pytest.mark.parametrize('change', [{'source':'other'}, {'page':99},
                                  {'text':'使用式（1.2）虚构来源'}, {'id':'missing'}])
def test_anchor_identity_page_and_original_text_are_required(change):
    added,omitted=evidence_companions([{**A,**change}],[A,F,T],[P])
    assert not added and omitted[0]['reason']=='invalid_or_filtered_anchor'


def test_integer_false_formula_and_text_are_excluded():
    assert not evidence_companions([A],[A,{**F,'indexable':0},T],[P])[0]
    assert [b['id'] for b in evidence_companions([A],[A,F,{**T,'indexable':0}],[P])[0]]==['f']
