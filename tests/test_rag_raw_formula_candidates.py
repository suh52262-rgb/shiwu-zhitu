from app.rag.raw_formula_candidates import raw_formula_candidates
import pytest


def fact(identity, indexable=1, generated=0):
    return {'id': identity, 'doc_id': 'book', 'page': 1, 'block_index': 2,
            'type': 'formula', 'text': 'original equation', 'indexable': indexable, 'is_generated': generated}


def test_raw_rank_is_preserved_and_not_mislabelled_retained():
    points = [{'id': 'prose'}, {'id': 'formula', 'score': .4}]
    facts = [dict(fact('prose'), type='text'), fact('formula')]
    pool, meta = raw_formula_candidates(points, [], facts)
    assert meta['added_ids'] == ['formula']
    assert pool[0]['relation']['kind'] == 'recorded_raw_vector_formula'
    assert pool[0]['relation']['raw_rank'] == 2
    assert 'retained_rank' not in pool[0]['relation']


def test_whole_candidate_and_depth_bounds():
    points = [{'id': 'first'}, {'id': 'second'}]
    facts = [fact('first'), fact('second')]
    assert raw_formula_candidates(points, [], facts, depth=1)[1]['added_ids'] == ['first']
    assert raw_formula_candidates(points, [], facts, budget=2)[0] == []
    assert raw_formula_candidates(points, [], facts, max_added=1)[1]['added_ids'] == ['first']


def test_unknown_and_filtered_ids_do_not_become_material():
    points = [{'id': 'unknown'}, {'id': 'off'}, {'id': 'generated'}]
    output, meta = raw_formula_candidates(points, [], [fact('off', 0), fact('generated', generated=1)])
    assert output == []
    assert len(meta['omissions']) == 3


def test_duplicate_raw_identity_rejected():
    with pytest.raises(ValueError):
        raw_formula_candidates([{'id': 'same'}, {'id': 'same'}], [], [fact('same')])
