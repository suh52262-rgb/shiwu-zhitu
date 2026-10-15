from types import SimpleNamespace

from app.agent import graph
from app.models.student import StudentState
from app.services.rag_materials import fragments, render


def hit(identity, text, page=1, neighbour=None):
    body = text + (' ' + neighbour if neighbour else '')
    spans = [{'id':identity,'source':'book','page':page,'block_index':page,
              'context_start':0,'context_end':len(text)}]
    if neighbour:
        spans.append({'id':'shared','source':'book','page':10,'block_index':10,
                      'context_start':len(text)+1,'context_end':len(body)})
    return {'block_id':identity,'source':'book','page':page,'text':text,
            'context':body,'context_blocks':spans}


def test_legacy_reports_only_three_actually_sent_hits():
    hits=[hit(str(i),'完整命题。',i) for i in range(5)]
    assert [s['id'] for s in fragments(hits,selected_policy='legacy')['sources']] == ['0','1','2']


def test_same_total_budget_preserves_lower_rank_hit_and_deduplicates():
    hits=[hit(str(i),'命中命题。'*20,i,'共享邻块。'*40) for i in range(5)]
    plan=fragments(hits,selected_policy='hit_first_total_v1')
    assert plan['content_characters']<=1200
    assert [s['id'] for s in plan['sources'][:5]]==[str(i) for i in range(5)]
    assert sum(s['id']=='shared' for s in plan['sources'])<=1


def test_no_room_for_complete_sentence_is_an_explicit_omission():
    plan=fragments([hit('long','无断句内容'*1000)],selected_policy='hit_first_total_v1')
    assert plan['sources']==[]
    assert plan['omissions'][0]['reason']=='no_complete_fragment_in_budget'


def test_invalid_span_is_omitted_without_inventing_source():
    row=hit('x','事实。')
    row['context_blocks'][0]['context_end']=9999
    plan=fragments([row],selected_policy='legacy')
    assert not plan['sources']
    assert plan['omissions'][0]['reason']=='invalid_context_interval'


def test_ui_sources_and_tutor_input_share_the_actual_plan(monkeypatch):
    monkeypatch.setenv('RAG_MATERIAL_POLICY','hit_first_total_v1')
    captured=[]
    def chat(system,user,schema):
        captured.append(user)
        return SimpleNamespace(value=SimpleNamespace(hint='请核对条件'), degraded=False)
    monkeypatch.setattr(graph,'get_llm',lambda _:SimpleNamespace(chat_structured_resilient=chat))
    hits=[hit(str(i),f'第{i}条真实完整命题。',i) for i in range(5)]
    graph.generate_hint(StudentState(hint_level=2,current_answer='当前书写',retrieved_context=hits))
    directory=graph.delivered_rag_sources(hits)
    assert len(directory)==5
    assert render({'sources':directory}) in captured[0]
