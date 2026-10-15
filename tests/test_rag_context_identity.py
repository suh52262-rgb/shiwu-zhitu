from app.rag import retriever,block_store

def b(i,text,index,**extra):
    return dict(id=i,page=1,block_index=index,type='text',text=text,indexable=True,is_generated=False,**extra)
def setup(monkeypatch,blocks):
    monkeypatch.setattr(block_store,'load_page_blocks',lambda doc,pages:{('doc',1):blocks})

def test_duplicate_text_uses_selected_identity_and_its_actual_neighbours(monkeypatch):
    text='这个重复段落包含相同的教材原文。'
    blocks=[b(str(i),'早段落'+str(i)+'：不同的上下文内容。',i) for i in range(12)]
    blocks[0]=b('first',text,0);blocks[10]=b('chosen',text,10)
    setup(monkeypatch,blocks);h={'source':'doc','page':1,'text':text,'block_id':'chosen'}
    retriever._expand_context([h],context_policy='legacy')
    assert h['context_blocks'][0]['id']=='chosen'
    assert '早段落8' in h['context'] and '早段落1：' not in h['context']

def test_known_id_never_falls_back_to_another_equal_text(monkeypatch):
    setup(monkeypatch,[b('other','确实相同的原始文字内容。',0)])
    h={'source':'doc','page':1,'text':'确实相同的原始文字内容。','block_id':'missing'}
    retriever._expand_context([h]);assert h['context_status']=='not_expanded' and 'context_blocks' not in h

def test_changed_text_and_filtered_anchor_do_not_expand(monkeypatch):
    for block,reason in [(b('id','修订后的真实教材原文。',0),'hit_text_changed'),({**b('id','原始内容包含很多文字。',0),'indexable':False},'hit_fact_filtered')]:
        setup(monkeypatch,[block]);h={'source':'doc','page':1,'text':'原始内容包含很多文字。','block_id':'id'}
        retriever._expand_context([h]);assert h['context_reason']==reason

def test_missing_id_duplicate_text_is_ambiguous(monkeypatch):
    text='缺少稳定身份时不能猜测位置。';setup(monkeypatch,[b('a',text,0),b('b',text,1)])
    h={'source':'doc','page':1,'text':text};retriever._expand_context([h])
    assert h['context_status']=='not_expanded'

def test_reexpansion_removes_old_policy_candidates_and_context(monkeypatch):
    text='新的原文只能补充新的上下文。';setup(monkeypatch,[b('a',text,0)])
    h={'source':'doc','page':1,'text':text,'block_id':'a','context':'过期内容','context_policy':'formula_refs_v2','context_link_candidates':[{'id':'old'}]}
    retriever._expand_context([h],context_policy='legacy')
    assert h['context']==text and 'context_policy' not in h and 'context_link_candidates' not in h
