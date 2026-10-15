from app.agent import graph
from app.agent.schemas import StepAnalysis
from app.models.student import StudentState
from fakes import install_fake_llm


def test_correct_but_incomplete_help_reaches_retrieval_once(monkeypatch):
    install_fake_llm(monkeypatch, analysis=StepAnalysis(current_step='概念说明',method='物理解释',
                     has_obvious_error=False,is_complete=False,concepts=[]))
    calls=[]
    def retrieve(**kwargs):
        calls.append(kwargs)
        return [{'text':'磁力垂直于速度，不做功。','source':'book','page':1,'block_id':'b1'}]
    monkeypatch.setattr(graph,'retrieve_knowledge_tool',retrieve)
    result=graph.tutor_graph.invoke(StudentState(current_answer='磁力不做功。',problem_content='解释原因',request_hint=True).model_dump())
    assert len(calls)==1
    assert '解释原因' in calls[0]['query'] and '磁力不做功' in calls[0]['query']
    assert result['rag_status']['state']=='ok'
    assert result['rag_materials_in_prompt']
    assert len(result['hint_history'])==1, 'retrieval must not execute hint policy twice'


def test_old_context_is_not_reused_in_a_correct_no_help_round(monkeypatch):
    install_fake_llm(monkeypatch,analysis=StepAnalysis(current_step='完成',method='计算',has_obvious_error=False,is_complete=True))
    monkeypatch.setattr(graph,'retrieve_knowledge_tool',lambda **kw: (_ for _ in ()).throw(AssertionError('unexpected retrieval')))
    result=graph.tutor_graph.invoke(StudentState(current_answer='结果与完整推导',problem_content='题',
        retrieved_context=[{'text':'OLD MATERIAL'}],rag_status={'state':'ok'},rag_materials_in_prompt=True).model_dump())
    assert result['retrieved_context']==[]
    assert result['rag_status']['state']=='not_triggered'
    assert not result['rag_materials_in_prompt']


def test_real_policy_completion_response_is_optional_and_records_actual_prompt(monkeypatch):
    install_fake_llm(monkeypatch,analysis=StepAnalysis(current_step='完成',method='计算',has_obvious_error=False,is_complete=True))
    monkeypatch.setattr(graph,'retrieve_knowledge_tool',lambda **kw:[{'text':'教材内容','source':'book','page':1}])
    result=graph.tutor_graph.invoke(StudentState(current_answer='完整解',problem_content='题',request_hint=True).model_dump())
    assert result['rag_status']['state']=='ok'
    assert '可选' in result['hint']
    assert result['hint_level']==2, 'first explicit help follows real policy, not a synthetic L1'
    assert result['hint_optional']
    assert result['rag_materials_in_prompt']
