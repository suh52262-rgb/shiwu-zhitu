"""Experimental same-document equation candidates. Not wired into live retrieval."""
from app.rag.formula_links import _REF, labels

def document_reference_candidates(anchors, facts, *, limit=2):
    """Resolve explicit references only; retain ambiguity and filtering failures.

    Facts are authoritative blocks with doc_id/source and stable IDs. The caller
    supplies already accepted candidate text as anchors, never evaluation gold.
    No page radius, inferred formula, unnumbered-neighbour or cross-book fallback.
    """
    if limit < 0:
        raise ValueError('Negative reference limit')
    existing = {a.get('id') for a in anchors}
    requests = {}
    for a in anchors:
        if not a.get('id') or not a.get('source') or a.get('type') != 'text':
            continue
        for x, y in _REF.findall(a.get('text') or ''):
            requests.setdefault((a['source'], f'{x}.{y}'), []).append(a['id'])
    added, omissions = [], []
    for (doc, number), anchor_ids in requests.items():
        matches = [b for b in facts if (b.get('doc_id') or b.get('source')) == doc
                   and b.get('type') == 'formula' and number in labels(b)]
        if len(matches) != 1 or not matches[0].get('id'):
            omissions.append({'source': doc, 'reference': number, 'reason': 'ambiguous_or_missing'})
            continue
        b = matches[0]
        if b.get('indexable') is False or b.get('is_generated') or not (b.get('text') or '').strip():
            omissions.append({'source': doc, 'reference': number, 'reason': 'filtered'})
            continue
        if b['id'] in existing:
            continue
        if len(added) >= limit:
            omissions.append({'source': doc, 'reference': number, 'reason': 'limit'})
            continue
        added.append({**b, 'source': doc, 'context_link': {
            'kind': 'experimental_document_equation_reference',
            'reference': number, 'anchor_ids': list(dict.fromkeys(anchor_ids))}})
        existing.add(b['id'])
    return added, omissions
