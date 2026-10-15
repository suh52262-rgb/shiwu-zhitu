import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rag_stage_attribution import classify

T = dict(id='f', doc_id='book', page=1, text='x=y')


def call(**changes):
    args = dict(indexed_ids={'f'}, raw_ids=[], kept_ids=[], returned_ids=[],
                loaded_pages=set(), catalogue=[], delivered=[])
    args.update(changes)
    return classify(T, **args)


@pytest.mark.parametrize('kwargs,expected', [
    ({}, 'outside_filtered_vector_top_n'),
    ({'indexed_ids': set()}, 'not_indexed_in_snapshot'),
    ({'raw_ids': ['f']}, 'outside_kept_top_k_or_deduplicated'),
    ({'kept_ids': ['f']}, 'outside_tool_return_top_k'),
    ({'loaded_pages': {('book', 1)}}, 'context_loaded_but_expansion_omitted'),
    ({'returned_ids': ['f']}, 'returned_hit_not_catalogued'),
    ({'catalogue': [{'id': 'f'}]}, 'material_selection_or_budget_excluded'),
    ({'delivered': [{'id': 'f', 'source': 'book', 'page': 1, 'text': 'x=y', 'source_start': 100, 'source_end': 103}]}, 'delivered_whole_block'),
    ({'delivered': [{'id': 'f', 'source': 'book', 'page': 1, 'text': 'x'}]}, 'delivered_partial_block'),
    ({'delivered': [{'id': 'f', 'source': 'book', 'page': 1}]}, 'unverifiable_delivery_text'),
    ({'delivered': [{'id': 'f', 'source': 'other', 'page': 1, 'text': 'x=y'}]}, 'conflicting_delivery_source'),
    ({'raw_ids': [None]}, 'unverifiable_stage_identity'),
    ({'eligible': False}, 'filtered_or_document_unavailable'),
])
def test_actual_stage_membership_and_range_are_distinct(kwargs, expected):
    assert call(**kwargs) == expected


def test_missing_target_identity_is_not_counted_as_no_recall():
    assert classify({'id': None}, indexed_ids=set(), raw_ids=[], kept_ids=[],
                    returned_ids=[], loaded_pages=set(), catalogue=[], delivered=[]) == 'unverifiable_identity'
