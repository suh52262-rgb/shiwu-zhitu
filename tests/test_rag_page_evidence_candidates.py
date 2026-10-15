import math
import pytest
from app.rag.page_evidence_candidates import page_evidence_candidates

H={'source':'book','page':1,'block_id':'a'}
A={'id':'a','doc_id':'book','page':1,'block_index':0,'type':'text','text':'这是一个已知的正文锚点。','bbox':[0,100,100,120]}
T={**A,'id':'t','block_index':5,'text':'适用条件必须在原文中真实存在。','bbox':[0,70,100,90]}
F={**A,'id':'f','block_index':20,'type':'formula','text':r'$$\omega^2=k/m$$','bbox':[]}

def test_same_page_context_preserves_text_without_inventing_dependencies():
    added,_=page_evidence_candidates([{**A,'source':'book'}],[H],[A,T,F])
    assert {x['id'] for x in added}=={'t','f'}
    assert next(x for x in added if x['id']=='f')['text']==F['text']
    assert all(x['context_relation']=={'kind':'same_page_only','anchor_id':'a','derivation_inferred':False} for x in added)

def test_wrong_document_page_generated_and_filtered_never_enter():
    facts=[A,{**T,'doc_id':'other'},{**F,'page':2},{**T,'id':'bad','indexable':False},{**F,'id':'gen','is_generated':True}]
    assert not page_evidence_candidates([{**A,'source':'book'}],[H],facts)[0]

def test_missing_or_invalid_positions_do_not_create_text_associations():
    for box in [[],[0,0,0,2],[0,math.nan,2,3]]:
        assert not page_evidence_candidates([{**A,'source':'book'}],[H],[A,{**T,'bbox':box}])[0]
    assert not page_evidence_candidates([], [{**H,'block_id':'missing'}], [A,T])[0]

def test_whole_block_budget_never_truncates_math_or_text():
    added,omitted=page_evidence_candidates([{**A,'source':'book'}],[H],[A,T,F],text_budget=1,formula_budget=1)
    assert not added and len(omitted)==2
    with pytest.raises(ValueError):page_evidence_candidates([],[],[],max_pages=-1)

def test_ambiguous_identity_and_caption_are_excluded():
    facts=[A,T,{**T,'text':'另一版本的段落内容'}, {**T,'id':'caption','type':'figure_caption'},{**F,'id':'tag','text':'(1.2)'}]
    assert not page_evidence_candidates([{**A,'source':'book'}],[H],facts)[0]

def test_distinct_page_limit_and_nearby_text_order():
    other={**T,'id':'other','page':2}
    far={**T,'id':'far','bbox':[0,1000,100,1020]}
    added,_=page_evidence_candidates([{**A,'source':'book'}],[H,{**H,'page':2,'block_id':'other'}],[A,T,far,other],max_pages=1)
    assert [x['id'] for x in added]==['t','far']
