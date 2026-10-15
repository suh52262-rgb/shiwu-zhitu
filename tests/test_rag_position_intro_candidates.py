import pytest
from copy import deepcopy
from app.rag.position_intro_candidates import position_intro_candidates

A={'id':'a','doc_id':'book','page':1,'type':'text','text':'允许的波长为','bbox':[10,10,150,20],'indexable':1}
F={'id':'f','doc_id':'book','page':1,'type':'formula','text':r'x=y\tag{1.2}','indexable':1}
P={'source':'book','page':1,'formula_id':'f','label_id':'label','reference':'1.2','label_bbox':[300,35,350,45],'result':{'status':'verified'}}
W={'source':'book','page':1,'single_column_body_observed':True}


def run(facts=None,p=None,verified=None,w=None,**kw):
    return position_intro_candidates([],facts or [A,F],p or [P],verified if verified is not None else ['a'],w if w is not None else [W],**kw)


def test_pair_preserves_original_text_and_never_claims_support():
    before=deepcopy([A,F]);added,_=run();assert [b['id'] for b in added]==['a','f'] and [A,F]==before
    assert not added[0]['relation']['semantic_support_validated']


@pytest.mark.parametrize('change',[{'doc_id':'other'},{'page':2},{'indexable':0},{'is_generated':True},
                                 {'type':'figure_caption'},{'bbox':[10,1,150,2]},{'text':'允许波长。'}])
def test_wrong_source_filter_type_gap_and_phrase_do_not_link(change):
    assert not run([{**A,**change},F])[0]


def test_missing_original_or_column_witness_blocks_link():
    assert not run(verified=[])[0] and not run(w=[])[0]
    assert not run(p=[{**P,'result':{'status':'rejected'}}])[0]


def test_intervening_caption_and_duplicate_formula_block_link():
    assert not run([A,F,{**A,'id':'caption','type':'figure_caption','bbox':[10,22,150,30]}])[0]
    assert not run([A,F,{**F,'id':'other'}])[0]


def test_pair_budget_never_returns_only_intro_or_only_formula():
    assert not run(budget=len(A['text'])+len(F['text'])-1)[0]
    assert not run(limit=1)[0]


def test_wrong_printed_reference_or_unobserved_two_column_layout_is_rejected():
    assert not run(p=[{**P,'reference':'1.3'}])[0]
    assert not run(w=[{**W,'single_column_body_observed':False}])[0]


def test_equally_close_verified_intros_are_ambiguous():
    second={**A,'id':'other'}
    assert not run([A,second,F],verified=['a','other'])[0]


def test_truncated_existing_intro_cannot_silently_support_new_formula():
    pool=[dict(A,source='book',text=A['text'][:2])]
    added,omitted=position_intro_candidates(pool,[A,F],[P],['a'],[W])
    assert not added and omitted[0]['reason']=='existing_pair_source_incomplete_or_conflicting'


def test_duplicate_baseline_identity_is_rejected():
    with pytest.raises(ValueError):position_intro_candidates([dict(A,source='book')]*2,[A,F],[P],['a'],[W])
