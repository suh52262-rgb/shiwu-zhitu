"""Experimental source-preserving prose completion, not semantic dependency links.

Each accepted block must have been checked against its original PDF text/box.
Never cross pages, figures, formulas, filtering barriers, new indentation or gaps.
Physical continuity is a candidate relationship only; sentence starts/subsections
remain unknown rather than inventing a complete paragraph or applicable condition.
"""
import math
import re

_END = re.compile(r'[。！？][”’」』）)]*\s*$')


def _box(b):
    value = b.get('bbox')
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value)
            and value[2] > value[0] and value[3] > value[1])


def _prose(b, verified):
    text = b.get('text') or ''
    return (b.get('id') in verified and b.get('type') == 'text'
            and b.get('indexable') not in (False, 0) and not b.get('is_generated')
            and _box(b) and b['bbox'][2]-b['bbox'][0] >= 80
            and re.search(r'[\u4e00-\u9fff]', text)
            and not re.match(r'^\s*(?:第.+章|\d+[.\-]\d+\s+\S)', text))


def _follows(a, b):
    x, y = a['bbox'], b['bbox']
    gap = y[1]-x[3]
    overlap = max(0, min(x[2], y[2])-max(x[0], y[0]))
    # A rightward indent is a new paragraph, never a continuation.
    return (0 <= gap <= 12 and -30 <= y[0]-x[0] <= 5
            and overlap/min(x[2]-x[0], y[2]-y[0]) >= .85)


def prose_companions(pool, facts, verified_ids, *, budget=1600, limit=8):
    if budget < 0 or limit < 0:
        raise ValueError('Negative completion limit')
    pages, index, conflicts = {}, {}, set()
    for b in facts:
        identity = b.get('id')
        if not identity:
            continue
        if identity in index:
            conflicts.add(identity)
            continue
        index[identity] = b
        if _box(b):
            pages.setdefault((b.get('doc_id') or b.get('source'), b.get('page')), []).append(b)
    for blocks in pages.values():
        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0], b['id']))
    output = [dict(s) for s in pool]
    existing = {s['id']: n for n, s in enumerate(output) if s.get('id')}
    spans, omissions, used, changed = [], [], 0, 0
    verified = set(verified_ids)-conflicts
    for seed in pool:
        b = index.get(seed.get('id'))
        if not b or (b.get('doc_id') or b.get('source'), b.get('page')) != (seed.get('source'), seed.get('page')):
            continue
        if not _prose(b, verified):
            continue
        if not b['text'].startswith(seed.get('text') or ''):
            omissions.append(dict(id=b['id'], reason='source_prefix_mismatch'))
            continue
        blocks = pages[(seed['source'], seed['page'])]
        i = next(n for n, item in enumerate(blocks) if item['id'] == b['id'])
        start, end = i, i
        while start > 0 and end-start < 5:
            previous, current = blocks[start-1], blocks[start]
            if not _prose(previous, verified) or _END.search(previous['text']) or not _follows(previous, current):
                break
            start -= 1
        while end+1 < len(blocks) and end-start < 5 and not _END.search(blocks[end]['text']):
            current, following = blocks[end], blocks[end+1]
            if not _prose(following, verified) or not _follows(current, following):
                break
            end += 1
        group = blocks[start:end+1]
        replacements = [g for g in group if g['id'] not in existing
                        or len(g['text']) > len(output[existing[g['id']]]['text'])]
        extra = sum(len(g['text'])-(len(output[existing[g['id']]]['text']) if g['id'] in existing else 0)
                    for g in replacements)
        if not replacements:
            continue
        if used+extra > budget or changed+len(replacements) > limit:
            omissions.append(dict(id=b['id'], reason='whole_span_budget_or_limit'))
            continue
        relation = dict(kind='geometric_prose_span', anchor_id=b['id'],
                        ordered_ids=[g['id'] for g in group], start_complete='unknown',
                        end_complete=bool(_END.search(group[-1]['text'])),
                        subsection_relation='unknown', semantic_support_validated=False)
        for g in replacements:
            source = dict(g, source=seed['source'], truncated=False, source_start=0,
                          source_end=len(g['text']), relation=relation)
            if g['id'] in existing:
                output[existing[g['id']]] = source
            else:
                existing[g['id']] = len(output)
                output.append(source)
        spans.append(relation)
        used += extra
        changed += len(replacements)
    return output, dict(extra_characters=used, changed_blocks=changed, spans=spans, omissions=omissions)
