from scripts.rag_candidate_trace import trace_block

FACT = {'id': 'a', 'source': 'book', 'page': 2, 'text': '完整正文', 'indexable': True}

def test_complete_prefix_and_source_drift_are_separate():
    assert trace_block(FACT, [], [FACT]) == 'candidate_complete'
    assert trace_block(FACT, [], [{**FACT, 'text': '完整'}]) == 'candidate_prefix_only'
    assert trace_block(FACT, [], [{**FACT, 'text': '另一个版本'}]) == 'source_drift'

def test_identity_missing_or_wrong_origin_is_not_a_retrieval_failure():
    assert trace_block({**FACT, 'id': None}, [], []) == 'unverifiable_identity'
    assert trace_block(FACT, [], [{**FACT, 'source': 'another-book'}]) == 'identity_conflict'

def test_saved_context_and_direct_hit_are_traced_without_claiming_rank():
    hit = {'source': 'book', 'context_blocks': [{'id': 'a'}]}
    assert trace_block(FACT, [hit], []) == 'saved_context_present_catalogue_excluded'
    assert trace_block(FACT, [{'source': 'book', 'block_id': 'a'}], []) == 'direct_hit_context_absent'
    assert trace_block(FACT, [], []) == 'absent_from_saved_context'

def test_filter_violation_and_legitimate_filtering_differ():
    filtered = {**FACT, 'indexable': False}
    assert trace_block(filtered, [], []) == 'filtered_fact'
    assert trace_block(filtered, [], [FACT]) == 'candidate_filter_violation'

def test_another_document_context_cannot_prove_presence():
    assert trace_block(FACT, [{'source': 'other', 'context_link_candidates': [{'id': 'a'}]}], []) == 'absent_from_saved_context'
