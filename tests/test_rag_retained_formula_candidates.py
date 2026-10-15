from copy import deepcopy
import pytest
from app.rag.retained_formula_candidates import retained_formula_candidates

F={'id':'f','doc_id':'book','page':1,'type':'formula','text':r'P=I^2R','indexable':1,'is_generated':0}
H={'block_id':'f','source':'book','page':1,'type':'formula','text':F['text']}


def test_only_actual_retained_identity_is_added_without_semantic_or_geometry_claim():
    before=deepcopy([F,H]);out,meta=retained_formula_candidates([H],[],[F])
    assert out[0]['text']==F['text'] and meta['added_ids']==['f'] and [F,H]==before
    assert not out[0]['relation']['formula_geometry_validated']
    assert retained_formula_candidates([],[],[F])[0]==[]


@pytest.mark.parametrize('change',[{'source':'other'},{'page':2},{'text':'P=wrong'},{'block_id':None}])
def test_wrong_retained_source_is_not_repaired(change):
    out,meta=retained_formula_candidates([{**H,**change}],[],[F]);assert not out and meta['omissions']


@pytest.mark.parametrize('change',[{'indexable':0},{'is_generated':True},{'type':'text'}])
def test_filtered_or_different_fact_cannot_enter(change):
    assert not retained_formula_candidates([H],[],[{**F,**change}])[0]


def test_whole_block_budget_and_baseline_identity_are_preserved():
    assert not retained_formula_candidates([H],[],[F],budget=len(F['text'])-1)[0]
    assert not retained_formula_candidates([H],[],[F],max_added=0)[0]
    pool=[dict(F,source='book')];assert retained_formula_candidates([H],pool,[F])[0]==pool
    with pytest.raises(ValueError):retained_formula_candidates([H],pool*2,[F])
