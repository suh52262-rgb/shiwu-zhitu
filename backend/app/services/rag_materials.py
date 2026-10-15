"""One source catalogue for Tutor input and UI, without evaluation gold access."""
import os


def policy():
    from app.config import get_settings
    value = os.getenv('RAG_MATERIAL_POLICY', get_settings().rag_material_policy)
    if value not in {'legacy', 'hit_first_total_v1', 'selected_evidence_v3'}:
        raise ValueError('Unknown RAG material policy')
    return value


def fragments(hits, per_hit_budget=400, selected_policy=None):
    selected_policy = selected_policy or policy()
    if selected_policy == 'selected_evidence_v3':
        from app.services.rag_selection import cached_plan
        return cached_plan(hits, per_hit_budget=per_hit_budget)
    if selected_policy not in {"legacy", "hit_first_total_v1", "catalogue"}:
        raise ValueError("Unknown material policy")
    candidates, omissions = [], []
    selected_hits = hits[:3] if selected_policy == 'legacy' else hits
    for rank, item in enumerate(selected_hits, 1):
        body = str(item.get('context') or item.get('text') or '')
        spans = item.get('context_blocks') or [{
            'id': item.get('block_id'), 'source': item.get('source'),
            'page': item.get('page'), 'block_index': item.get('block_index'),
            'type': item.get('type'), 'context_start': 0, 'context_end': len(body)}]
        for position, span in enumerate(spans):
            start, end = int(span.get('context_start', 0)), int(span.get('context_end', 0))
            if not 0 <= start < end <= len(body):
                omissions.append({'id': span.get('id'), 'reason': 'invalid_context_interval'})
                continue
            limit = min(end, per_hit_budget) if selected_policy == 'legacy' else end
            if start >= limit:
                omissions.append({'id': span.get('id'), 'reason': 'per_hit_budget'})
                continue
            text = body[start:limit].strip()
            if not text:
                continue
            candidates.append({
                'id': span.get('id'), 'source': span.get('source') or item.get('source'),
                'page': span.get('page', item.get('page')), 'block_index': span.get('block_index'),
                'type': span.get('type'), 'text': text, 'truncated': limit < end,
                'hit_rank': rank, 'context_position': position,
                'direct': str(span.get('id')) == str(item.get('block_id')) and bool(span.get('id')),
                'source_start': start, 'source_end': limit})
            if span.get('context_link'):
                candidates[-1]['context_link'] = span['context_link']
    if selected_policy == 'catalogue':
        return {'sources':candidates,'omissions':omissions,'policy':'catalogue'}
    if selected_policy == 'legacy':
        return _append_formula_links({'policy': selected_policy, 'sources': candidates, 'omissions': omissions,
                'content_characters': sum(len(c['text']) for c in candidates)}, hits, 3*per_hit_budget)
    # Direct hits across the existing candidates precede supporting neighbours.
    # No widening of retrieval depth, model calls or reading the labelled gold.
    candidates.sort(key=lambda c: (not c['direct'], c['hit_rank'], c['context_position']))
    sources, seen, used = [], set(), 0
    total_budget = 3 * per_hit_budget
    for candidate in candidates:
        identity = (candidate['source'], candidate['page'], candidate['block_index'], candidate['id'])
        if candidate['id'] is None and candidate['block_index'] is None:
            identity += (candidate['text'],)
        if identity in seen:
            continue
        seen.add(identity)
        remaining = total_budget - used
        text = candidate['text']
        if len(text) > remaining:
            # Preserve full sentences/lines where possible; record all omissions.
            prefix = text[:remaining]
            boundary = max(prefix.rfind('。'), prefix.rfind('\n'), prefix.rfind('；')) + 1
            if boundary:
                text = prefix[:boundary]
            else:
                omissions.append({'id': candidate['id'], 'reason': 'no_complete_fragment_in_budget'})
                continue
        if not text:
            omissions.append({'id': candidate['id'], 'reason': 'total_budget'})
            continue
        truncated = len(text) < len(candidate['text'])
        sources.append({**candidate, 'text': text, 'truncated': truncated,
                        'source_end': candidate['source_start'] + len(text)})
        if truncated:
            omissions.append({'id': candidate['id'], 'reason': 'partial_total_budget'})
        used += len(text)
    return _append_formula_links({'policy': selected_policy, 'sources': sources, 'omissions': omissions,
            'content_characters': used, 'total_budget': total_budget}, hits, total_budget)


def _append_formula_links(plan, hits, budget):
    """Candidate v2 preserves every baseline fragment; never borrows its budget.

    Duplicate prefixes of one verified source may share one full fragment.
    No identity means no deduplication. Whole formulas only, with omissions.
    """
    if not any(h.get('context_policy') == 'formula_refs_v2' for h in hits):
        return plan
    sources=[]
    for source in plan['sources']:
        same=next((s for s in sources if source.get('id') and
                   (s['source'],s['page'],s['id'])==(source['source'],source['page'],source['id'])
                   and (s['text'].startswith(source['text']) or source['text'].startswith(s['text']))),None)
        if same is None:sources.append(dict(source))
        elif len(source['text'])>len(same['text']):same.update(source)
    used=sum(len(s['text']) for s in sources)
    omissions=list(plan['omissions'])
    for rank,hit in enumerate(hits,1):
        if hit.get('context_policy')!='formula_refs_v2':continue
        for b in hit.get('context_link_candidates',[]):
            if not b.get('id') or not b.get('text'):continue
            same=next((s for s in sources if (s['source'],s['page'],s['id'])==
                       (hit['source'],b['page'],b['id'])),None)
            if same and not same['truncated']:continue
            if same and not b['text'].startswith(same['text']):
                omissions.append({'id':b['id'],'reason':'formula_source_conflict'});continue
            cost=len(b['text'])-(len(same['text']) if same else 0)
            if used+cost>budget:
                omissions.append({'id':b['id'],'reason':'no_spare_budget_for_complete_formula'});continue
            linked={'id':b['id'],'source':hit['source'],'page':b['page'],'block_index':b['block_index'],
                    'type':'formula','text':b['text'],'truncated':False,'hit_rank':rank,
                    'context_position':None,'direct':False,'source_start':0,'source_end':len(b['text']),
                    'context_link':b['context_link']}
            if same:same.update(linked)
            else:sources.append(linked)
            used+=cost
    assert all(any(s['id']==old['id'] and s['source']==old['source'] and s['page']==old['page']
                   and old['text'] in s['text'] for s in sources) for old in plan['sources'])
    return {**plan,'sources':sources,'omissions':omissions,'content_characters':used,
            'total_budget':budget,'context_link_policy':'formula_refs_v2','baseline_fragments_preserved':True}


def render(plan):
    return '\n'.join(f"- 「{s['text']}」（{s['source']} 第{s['page']}页"
                     + ('，**该页此处被截断**' if s['truncated'] else '') + '）'
                     for s in plan['sources'])
