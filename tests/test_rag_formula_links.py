from copy import deepcopy
import pytest
from app.rag.formula_links import linked_formulas
from app.rag import retriever, block_store
from app.services.rag_materials import fragments

def block(identity,text,kind='text',page=2,**kw):
    return dict(id=identity,text=text,type=kind,page=page,block_index=0,
                indexable=True,is_generated=False,**kw)

def test_unique_previous_page_and_explicit_only():
    anchor=block('a','由（1.79）式得到储能。')
    formula=block('f',r'W=CU^2/2 \tag{1.79}','formula',1)
    assert linked_formulas([anchor],[formula],hit_page=2)[0][0]['id']=='f'
    assert linked_formulas([block('a','电容器储能')],[formula],hit_page=2)[0]==[]
    assert linked_formulas([anchor],[{**formula,'page':3}],hit_page=2)[0]==[]

@pytest.mark.parametrize('changes',[{'indexable':False},{'is_generated':True}])
def test_filter_cannot_be_bypassed(changes):
    a=block('a','式（1.2）说明了一个关系。')
    f=block('f',r'x=y \tag{1.2}','formula')
    assert linked_formulas([a],[{**f,**changes}],hit_page=2)[0]==[]
    assert linked_formulas([{**a,**changes}],[f],hit_page=2)[0]==[]

def test_ambiguous_missing_id_and_unnumbered_are_not_guessed():
    a=block('a','式（1.2）说明了一个关系。')
    f=block('f',r'x=y \tag{1.2}','formula')
    for pool in ([f,{**f,'id':'other'}],[{**f,'id':None}],[{**f,'text':'x=y'}]):
        assert linked_formulas([a],pool,hit_page=2)[0]==[]

def test_default_unchanged_and_candidate_uses_production_delivery(monkeypatch):
    a=block('a','利用公式（1.2）可以核对本题的物理关系。')
    f=block('f',r'x=y+z \tag{1.2}','formula',1)
    pool={('doc',2):[a],('doc',1):[f]}
    monkeypatch.delenv('RAG_CONTEXT_POLICY',raising=False)
    monkeypatch.setattr(block_store,'load_page_blocks',lambda doc,pages:deepcopy(pool))
    hit=dict(source='doc',page=2,text=a['text'],block_id='a')
    default=[deepcopy(hit)];explicit=[deepcopy(hit)];candidate=[deepcopy(hit)]
    retriever._expand_context(default)
    retriever._expand_context(explicit,context_policy='legacy')
    assert default==explicit
    retriever._expand_context(candidate,context_policy='formula_refs_v1')
    sources=fragments(candidate,selected_policy='legacy')['sources']
    assert sources[0]['id']=='a'
    support=next(s for s in sources if s['id']=='f')
    assert support['text']==f['text'] and support['page']==1
    assert support['context_link']['anchor_ids']==['a']
    assert len(candidate[0]['context'])<=retriever.CONTEXT_MAX_CHARS

def test_unknown_policy_rejected():
    with pytest.raises(ValueError):retriever._expand_context([],context_policy='unknown')

def test_hit_references_precede_neighbour_numbers():
    a=block('a','由（1.20）和（1.21）得到结果。')
    hit=block('hit','要核对（1.24）式。')
    formulas=[block(str(n),f'x=y \\tag{{1.{n}}}','formula') for n in (20,21,24)]
    result,_=linked_formulas([a,hit],formulas,hit_page=2,hit_id='hit')
    assert result[0]['context_link']['reference']=='1.24'

def test_v2_never_displaces_body_and_deduplicates_links(monkeypatch):
    a=block('a','由（1.2）式可以核对相关物理量的关系。')
    b=block('body','这段正文给出了公式适用条件，需要完整保留。')
    f=block('f',r'x=y+z \tag{1.2}','formula',1)
    monkeypatch.setattr(block_store,'load_page_blocks',lambda doc,pages:{('doc',2):[a,b],('doc',1):[f]})
    hits=[dict(source='doc',page=2,text=a['text'],block_id='a') for _ in range(2)]
    legacy=deepcopy(hits);candidate=deepcopy(hits)
    retriever._expand_context(legacy,context_policy='legacy')
    retriever._expand_context(candidate,context_policy='formula_refs_v2')
    assert [h['context'] for h in legacy]==[h['context'] for h in candidate]
    old=fragments(legacy,selected_policy='legacy');new=fragments(candidate,selected_policy='legacy')
    assert all(any(s['id']==v['id'] and v['text'] in s['text'] for s in new['sources']) for v in old['sources'])
    assert sum(s['id']=='f' for s in new['sources'])==1

def test_v2_insufficient_budget_does_not_truncate_formula_or_drop_body():
    from app.services.rag_materials import _append_formula_links
    body={'id':'a','source':'doc','page':2,'text':'需要保留的正文','truncated':False}
    f=block('f','very_long_formula','formula',1,context_link={'kind':'explicit_equation_reference'})
    hits=[{'source':'doc','context_policy':'formula_refs_v2','context_link_candidates':[f]}]
    p=_append_formula_links({'sources':[body],'omissions':[]},hits,len(body['text']))
    assert p['sources']==[body]
    assert p['omissions'][0]['reason']=='no_spare_budget_for_complete_formula'
