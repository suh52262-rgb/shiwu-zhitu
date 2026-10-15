"""Experimental unified allocation; not registered as a production policy.

Only existing candidate IDs can be selected. Failure preserves the caller's
baseline, with an explicit failure status; no partial budget success is hidden.
"""
from copy import deepcopy
import hashlib
import json
import time

from app.services.rag_selection import RULE, Selection, selection_request

JOINT_RULE = RULE + '\n编号引用或版面连续性只是候选关联，不能据此假定公式、解释与条件互相支持。请核对实际原文；预算内必须同时保留回答所需的关系式、符号说明及适用条件。不得把只提供了一部分的材料当成充分答案。'


def unified_select(pool, query, fallback, *, budget=1200, llm=None):
    if budget < 0:
        raise ValueError('Negative material budget')
    baseline = deepcopy(fallback)
    result = dict(baseline, policy='experimental_joint_v1', selection_status='failed',
                  total_budget=budget, semantic_support_validated=False)
    started = time.perf_counter()
    try:
        if sum(len(s['text']) for s in baseline.get('sources', [])) > budget:
            result.update(sources=[], content_characters=0, fallback_error='Fallback exceeds declared budget')
        byid = {}
        for source in pool:
            identity = source.get('id')
            if not identity or identity in byid or not source.get('source') or not source.get('page') or not source.get('text'):
                raise ValueError('Missing or duplicate candidate identity/source')
            byid[identity] = source
        result['candidate_fingerprint'] = hashlib.sha256(json.dumps(pool, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if not pool:
            result.update(sources=[], content_characters=0, selection_status='empty', selected_ids=[])
            return result
        if llm is None:
            from app.services.llm import get_llm
            llm = get_llm('tutor')
        request, aliases = selection_request(pool, query, budget, 'full_v1')
        raw = llm.chat_structured(JOINT_RULE, json.dumps(request, ensure_ascii=False), Selection)
        if len(raw.selected_ids) != len(set(raw.selected_ids)) or any(i not in aliases for i in raw.selected_ids):
            raise ValueError('Invalid selected identity')
        ids = [aliases[i] for i in raw.selected_ids]
        chosen = [deepcopy(byid[i]) for i in ids]
        used = sum(len(s['text']) for s in chosen)
        # Reject the whole proposal, rather than silently dropping its final
        # equation or condition and returning a deceptively successful subset.
        if used > budget:
            raise ValueError('Selected group exceeds total budget')
        if any(s['type']=='formula' and s.get('truncated') for s in chosen):
            raise ValueError('Selected formula is incomplete')
        result.update(sources=chosen, content_characters=used,
                      selection_status='ready' if chosen else 'empty', selected_ids=ids,
                      omissions=[dict(id=i, reason='not_selected') for i in byid if i not in ids])
    except Exception as error:
        result['selection_error'] = type(error).__name__+': '+str(error)
    finally:
        result['selection_seconds'] = time.perf_counter()-started
    return result
