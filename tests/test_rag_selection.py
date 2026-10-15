from copy import deepcopy
from types import SimpleNamespace
import json
from app.services.rag_selection import select,cached_plan
from app.services.rag_materials import fragments

def hits():
    return [{'source':'doc','page':1,'block_id':'text','text':'此处给出该公式的适用条件。',
             'context':'此处给出该公式的适用条件。','context_policy':'formula_refs_v2',
             'context_blocks':[{'id':'text','source':'doc','page':1,'type':'text','block_index':0,
                               'context_start':0,'context_end':len('此处给出该公式的适用条件。')}],
             'context_link_candidates':[{'id':'f','page':1,'type':'formula','block_index':9,
                                        'text':'E=mc^2','context_link':{'reference':'1.2'}}]}]

class Judge:
    def __init__(self,ids):self.ids=ids;self.calls=0
    def chat_structured(self,system,user,schema):
        self.calls+=1;self.request=json.loads(user)
        return schema(selected_ids=self.ids)

def test_only_original_ids_and_text_selected_and_no_gold_input():
    h=hits();j=Judge(['text','f']);p=select(h,'解释这个公式的条件',llm=j)
    assert p['selection_status']=='ready' and p['sources'][-1]['text']=='E=mc^2'
    assert set(j.request)=={'query','budget','candidates'}
    h[0]['_selected_material']=p
    assert cached_plan(h)['sources']==p['sources']
    assert j.calls==1

def test_invalid_duplicate_or_unknown_identity_falls_back():
    for ids in [['foreign'],['f','f']]:
        p=select(hits(),'问题',llm=Judge(ids))
        assert p['selection_status']=='failed' and not any(s['id']=='foreign' for s in p['sources'])

def test_formula_over_budget_omitted_whole():
    p=select(hits(),'问题',llm=Judge(['f']),budget=2)
    assert p['sources']==[] and p['omissions'][0]['reason']=='complete_fragment_exceeds_budget'

def test_stale_or_fabricated_cache_not_used():
    h=hits();p=select(h,'问题',llm=Judge(['f']));h[0]['_selected_material']=p
    h[0]['_selected_material']['sources'][0]['text']='模型编造的教材原文'
    assert cached_plan(h)['selection_status']=='not_prepared_or_stale'
    h=hits();h[0]['_selected_material']=select(h,'问题',llm=Judge(['f']))
    h[0]['context_link_candidates'][0]['text']='new version'
    assert cached_plan(h)['selection_status']=='not_prepared_or_stale'

def test_default_render_does_not_invoke_selector(monkeypatch):
    monkeypatch.setenv('RAG_MATERIAL_POLICY','hit_first_total_v1')
    h=hits();p=fragments(h)
    assert p['policy']=='hit_first_total_v1'

def test_production_tool_selects_once_and_render_reuses_it(monkeypatch):
    from app.agent import tools
    from app.rag import retriever,block_store,rerank,query_expansion
    import app.services.llm
    monkeypatch.setenv('RAG_MATERIAL_POLICY','selected_evidence_v3')
    monkeypatch.setattr(query_expansion,'ENABLED',False)
    source=hits()[0]
    blocks=[{'id':'text','text':source['text'],'page':1,'block_index':0,'type':'text','indexable':True},
            {'id':'f','text':r'E=mc^2 \tag{1.2}','page':1,'block_index':9,'type':'formula','indexable':True}]
    blocks[0]['text']+='由（1.2）式可得。'
    raw=[{'source':'doc','page':1,'block_id':'text','text':blocks[0]['text']}]
    monkeypatch.setattr(tools,'retrieve',lambda *a,**k:deepcopy(raw))
    monkeypatch.setattr(block_store,'load_page_blocks',lambda doc,pages:{('doc',1):deepcopy(blocks)})
    monkeypatch.setattr(rerank,'rerank',lambda query,results,**kw:results)
    j=Judge(['text','f']);monkeypatch.setattr(app.services.llm,'get_llm',lambda role:j)
    status={};result=tools.retrieve_knowledge('解释这个公式',status=status)
    assert j.calls==1 and status['material_selection_status']=='ready'
    assert fragments(result)['sources'][-1]['id']=='f'
    assert j.calls==1

def test_compact_transport_preserves_text_and_maps_ids():
    from app.services.rag_selection import catalogue,selection_request
    h=hits();pool=catalogue(h)
    req,mapping=selection_request(pool,'问题',1200,'compact_v2')
    assert [s['text'] for s in req['candidates']]==[s['text'] for s in pool]
    assert mapping['C2']=='f'
    p=select(h,'问题',llm=Judge(['C2']),input_format='compact_v2')
    assert p['selection_status']=='ready' and p['sources'][0]['id']=='f'
    assert p['raw_model_ids']==['C2'] and p['raw_selected_ids']==['f']

def test_compact_rejects_fabricated_alias_or_unknown_format():
    assert select(hits(),'问题',llm=Judge(['C99']),input_format='compact_v2')['selection_status']=='failed'
    assert select(hits(),'问题',llm=Judge([]),input_format='not-a-format')['selection_status']=='failed'
