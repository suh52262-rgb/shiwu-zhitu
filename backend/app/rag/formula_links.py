"""Numbered textbook formula links; no query or evaluation-gold access."""
import re

_REF = re.compile(r'[（(]\s*(\d+)\s*[.．]\s*(\d+)\s*[）)]')
_TAG = re.compile(r'\\tag\s*\{\s*(\d+)\s*[.．]\s*(\d+)\s*\}')

def labels(block):
    result = {f'{a}.{b}' for a,b in _TAG.findall(block.get('text') or '')}
    raw = str(block.get('formula_number') or '').strip()
    m = re.fullmatch(r'[（(]?\s*(\d+)\s*[.．]\s*(\d+)\s*[）)]?', raw)
    if m:result.add(f'{m[1]}.{m[2]}')
    return result

def linked_formulas(anchors, blocks, *, hit_page, include_generated=False, hit_id=None):
    """One caller-scoped document, same/previous page, unique identity only."""
    requested={}
    for a in anchors:
        if (not a.get('id') or a.get('type')!='text' or a.get('indexable') is False
                or a.get('is_generated') and not include_generated):continue
        for x,y in _REF.findall(a.get('text') or ''):
            requested.setdefault(f'{x}.{y}',[]).append(a['id'])
    links,omissions=[],[]
    order = list(requested) if hit_id else sorted(requested)
    if hit_id:
        order.sort(key=lambda number: hit_id not in requested[number])
    for number in order:
        matches=[b for b in blocks if b.get('type')=='formula'
                 and b.get('page') in {hit_page,hit_page-1} and number in labels(b)]
        ids={b.get('id') for b in matches}
        if len(ids)!=1 or None in ids:
            omissions.append({'reference':number,'reason':'ambiguous_or_missing_formula'});continue
        b=matches[0]
        if b.get('indexable') is False or b.get('is_generated') and not include_generated:
            omissions.append({'reference':number,'reason':'formula_filtered'});continue
        if len(links)>=2:
            omissions.append({'reference':number,'reason':'formula_link_limit'});continue
        links.append({**b,'context_link':{'kind':'explicit_equation_reference',
                     'reference':number,'anchor_ids':requested[number]}})
    return links,omissions
