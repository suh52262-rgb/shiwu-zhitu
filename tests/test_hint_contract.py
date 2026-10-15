from app.models.student import StudentState
from app.services.hint_contract import hint_contract, render_hint_contract


def test_intermediate_number_does_not_become_requested_quantity():
    state = StudentState(current_answer='1/2×2×5²=25', problem_content='求重力做功 W_G')
    contract = hint_contract(state)
    assert contract['current_writing'] == state.current_answer
    assert not contract['authored_result_fields']
    assert 'W_G=25' not in render_hint_contract(state)
    assert any('中间数' in value for value in contract['constraints'])


def test_contract_does_not_turn_completion_into_verified_physics():
    contract = hint_contract(StudentState(current_answer='v=3 m/s', is_complete=True))
    assert contract['completion_judgement']
    assert contract['completion_is_model_judgement']
    assert any('速率是标量' in value for value in contract['constraints'])
    assert any('可选' in value for value in contract['constraints'])


def test_authored_spans_kept_without_reassigning_or_grading():
    fields = [{'key': 'F', 'quote': 'F=49 N', 'present': True, 'start': 0, 'end': 6}]
    contract = hint_contract(StudentState(subquestion_context={'target':'3', 'result_fields':fields}))
    assert contract['authored_result_fields'] == fields
    assert contract['question_scope'] == '3'


def test_completed_correct_l1_help_is_optional_without_model_call(monkeypatch):
    from app.agent import graph
    monkeypatch.setattr(graph, 'get_llm', lambda *args: (_ for _ in ()).throw(AssertionError('unexpected model call')))
    state = StudentState(current_answer='v=3 m/s', is_complete=True, round_correct=True,
                         request_hint=True, hint_level=1)
    result = graph.generate_hint(state)
    assert result['hint_level'] == 1
    assert '可选' in result['hint']
    assert '补' not in result['hint'] and '再核对' not in result['hint']
    assert '非模型文本' in result['hint_reasons'][-1]
