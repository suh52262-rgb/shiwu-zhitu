from copy import deepcopy
import pytest
from app.rag.region_pool import merge_range_inventory

A={'id':'a','source':'book','page':1,'type':'text','text':'原文','truncated':True}
B={'id':'b','source':'book','page':2,'type':'text','text':'仅在这些条件下适用。','bbox':[0,10,100,20],
   'indexable':1,'relation':{'kind':'original_pdf_subsection_range','semantic_support_validated':False}}


def test_whole_text_merge_preserves_baseline_and_provenance():
    before=deepcopy([A,B]);out,meta=merge_range_inventory([A],[B])
    assert [A,B]==before and out[0]==A and out[1]['text']==B['text']
    assert not meta['actual_delivery_evaluated'] and not meta['semantic_support_validated']


def test_no_partial_block_or_hidden_baseline_drop_at_budget():
    out,meta=merge_range_inventory([A],[B],budget=len(A['text'])+len(B['text'])-1)
    assert out==[A] and meta['omissions'][0]['id']=='b'
    assert merge_range_inventory([A],[B],limit=1)[0]==[A]


def test_prefix_extension_retains_identity_and_clears_truncation():
    full={**B,'id':'a','page':1,'text':'原文完整段。'}
    out,meta=merge_range_inventory([A],[full])
    assert out[0]['id']=='a' and not out[0]['truncated'] and meta['extended_ids']==['a']


@pytest.mark.parametrize('change',[{'source':'other'},{'page':99},{'text':'不同原文'},{'indexable':0},
                                 {'is_generated':True},{'type':'figure_caption'}])
def test_conflicting_and_filtered_ranges_are_rejected(change):
    with pytest.raises(ValueError):merge_range_inventory([A],[{**B,'id':'a',**change}])


def test_duplicates_and_oversize_baseline_are_rejected():
    with pytest.raises(ValueError):merge_range_inventory([A,A],[])
    with pytest.raises(ValueError):merge_range_inventory([A],[B,B])
    with pytest.raises(ValueError):merge_range_inventory([A],[],budget=1)
