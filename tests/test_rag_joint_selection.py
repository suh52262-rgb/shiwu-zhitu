from copy import deepcopy
import json

import pytest

from app.services.rag_joint_selection import unified_select

POOL = [dict(id='t', source='book', page=1, type='text', text='这条关系只适用于理想条件。', truncated=False),
        dict(id='f', source='book', page=1, type='formula', text='x=y', truncated=False)]
FALLBACK = dict(sources=[POOL[0]], content_characters=len(POOL[0]['text']), policy='hit_first_total_v1')


class Model:
    def __init__(self, ids):
        self.ids = ids

    def chat_structured(self, system, user, schema):
        self.request = json.loads(user)
        return schema(selected_ids=self.ids)


def test_unified_budget_can_replace_old_material_and_server_copies_exact_source():
    before = deepcopy(POOL); model = Model(['f'])
    result = unified_select(POOL, '关系式是什么', FALLBACK, llm=model, budget=3)
    assert result['selection_status']=='ready' and result['sources']==[POOL[1]]
    assert result['content_characters']==3 and POOL==before
    assert set(model.request)=={'query', 'budget', 'candidates'}
    assert result['semantic_support_validated'] is False


@pytest.mark.parametrize('ids,budget', [(['f','t'],3), (['f','f'],1200), (['unknown'],1200)])
def test_invalid_or_over_budget_group_is_not_silently_truncated(ids, budget):
    fallback = FALLBACK if budget==1200 else dict(sources=[POOL[1]], content_characters=3)
    result = unified_select(POOL, '问题', fallback, llm=Model(ids), budget=budget)
    assert result['selection_status']=='failed' and result['sources']==fallback['sources']


def test_incomplete_formula_and_conflicting_identity_are_rejected():
    assert unified_select([{**POOL[1], 'truncated': True}], '问题', FALLBACK,
                          llm=Model(['f']))['selection_status']=='failed'
    assert unified_select(POOL+[POOL[0]], '问题', FALLBACK, llm=Model(['f']))['selection_status']=='failed'


def test_empty_no_extra_model_calls_and_negative_budget():
    result = unified_select([], '问题', FALLBACK, llm=Model(['unknown']))
    assert result['selection_status']=='empty' and result['sources']==[]
    assert unified_select(POOL, '问题', FALLBACK, llm=Model([]))['selection_status']=='empty'
    with pytest.raises(ValueError):
        unified_select(POOL, '问题', FALLBACK, budget=-1)


def test_oversized_fallback_cannot_be_returned_as_within_declared_budget():
    result = unified_select(POOL, '问题', FALLBACK, budget=2, llm=Model(['f']))
    assert result['selection_status']=='failed' and result['sources']==[]
