"""Visible writing constraints, without inferring correctness from a number."""
import json


def hint_contract(state):
    return {
        'version': 'visible_work_v1',
        'current_writing': state.current_answer,
        'question_scope': state.subquestion_context.get('target') or 'unresolved',
        'authored_result_fields': state.subquestion_context.get('result_fields', []),
        'completion_judgement': bool(state.is_complete),
        'completion_is_model_judgement': True,
        'constraints': [
            '数值出现不证明它属于所求物理量；不得替学生给中间数添加新的量名或答案对应关系。',
            '速率是标量，未问速度方向时不要求方向；力和速度按题目实际要求核对。',
            '已完成且无明确错误时简短确认已写内容，不重复索要已展示步骤；巩固练习须标为可选。',
            '缺少展示只可请求补充依据，不推断学生未理解或未计算。',
        ],
    }


def render_hint_contract(state):
    return '\n【提示约束（事实与推断分开）】\n' + json.dumps(hint_contract(state), ensure_ascii=False)
