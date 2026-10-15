"""Experimental recorded raw-vector formula preservation, never default search.

The recording contains IDs/scores, not source payloads. Bind identities to the
frozen corpus and keep original raw rank; do not call these retained hits.
"""
from copy import deepcopy


def raw_formula_candidates(points, pool, facts, *, depth=30, max_added=2, budget=12000, limit=48):
    if min(depth, max_added, budget, limit) < 0:
        raise ValueError('Negative candidate bound')
    byid = {b['id']: b for b in facts}
    existing = {b['id'] for b in pool}
    identities = [p.get('id') for p in points]
    if len(byid) != len(facts) or len(existing) != len(pool) or not all(identities) or len(set(identities)) != len(points):
        raise ValueError('Missing or duplicate source identity')
    output = deepcopy(pool)
    used = sum(len(b['text']) for b in output)
    if used > budget or len(output) > limit:
        raise ValueError('Baseline over budget')
    added, omissions = [], []
    for rank, point in enumerate(points[:depth], 1):
        identity = point['id']
        fact = byid.get(identity)
        if fact is None:
            omissions.append({'id': identity, 'reason': 'raw_identity_not_bound'})
            continue
        if fact.get('type') != 'formula' or identity in existing:
            continue
        if fact.get('indexable') not in (True, 1) or fact.get('is_generated') or not fact.get('text'):
            omissions.append({'id': identity, 'reason': 'raw_formula_filtered'})
            continue
        if len(added) >= max_added or len(output) >= limit or used + len(fact['text']) > budget:
            omissions.append({'id': identity, 'reason': 'whole_candidate_budget_or_limit'})
            continue
        source = dict(deepcopy(fact), source=fact['doc_id'], source_start=0,
                      source_end=len(fact['text']), truncated=False,
                      relation={'kind': 'recorded_raw_vector_formula', 'raw_rank': rank,
                                'raw_score': point.get('score'), 'semantic_support_validated': False,
                                'formula_geometry_validated': False})
        output.append(source)
        existing.add(identity)
        added.append(identity)
        used += len(fact['text'])
    return output, {'added_ids': added, 'omissions': omissions, 'candidate_characters': used,
                    'raw_depth': depth, 'actual_delivery_evaluated': False,
                    'semantic_support_validated': False}
