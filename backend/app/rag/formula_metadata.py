"""Conservative missing-number proposals; never mutate indexed facts.

Only simple ASCII equations identical to a unique original PDF text line qualify.
Geometry proves printed association, not physical meaning or subsection membership.
"""
import math
import re

from app.rag.formula_links import labels
from app.rag.formula_positions import standalone_label


def simple_equation(text):
    value = re.sub(r'\s+', '', (text or '').replace('$$', ''))
    if '=' not in value or not re.fullmatch(r'[A-Za-z0-9=+*/().,<>^_\-]+', value):
        return None
    return value


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value)
            and value[0] < value[2] and value[1] < value[3])


def recover_missing_numbers(facts, lines):
    """One complete document's facts and original PDF lines, each with page/bbox.

    Returns proposals and explicit omissions. Proposals require unique identity,
    unique document label, matching stored label box, and one formula to its left
    with >=60% vertical overlap. No block adjacency, OCR repair or LaTeX expansion.
    """
    documents = {b.get('doc_id') or b.get('source') for b in facts}
    if len(documents) != 1 or not next(iter(documents), None):
        raise ValueError('Exactly one identified document is required')
    proposals, omissions = [], []
    for anchor in facts:
        number = standalone_label(anchor.get('text'))
        if anchor.get('type') != 'text' or not number:
            continue
        identity = anchor.get('id')
        page = anchor.get('page')
        doc = anchor.get('doc_id') or anchor.get('source')
        reason = None
        same = [b for b in facts if b.get('type') == 'text'
                and standalone_label(b.get('text')) == number]
        if any(number in labels(b) for b in facts if b.get('type') == 'formula'):
            continue
        if not doc or not identity or page is None or len(same) != 1:
            reason = 'ambiguous_label_identity'
        elif anchor.get('indexable') in (False, 0) or anchor.get('is_generated'):
            reason = 'filtered_label'
        elif not _box(anchor.get('bbox')):
            reason = 'missing_label_position'
        label_lines = [x for x in lines if x.get('page') == page
                       and standalone_label(x.get('text')) == number]
        if not reason and (len(label_lines) != 1 or not _box(label_lines[0].get('bbox'))):
            reason = 'original_label_not_unique'
        if not reason:
            box = label_lines[0]['bbox']
            if max(abs(x-y) for x, y in zip(box, anchor['bbox'])) > 2:
                reason = 'label_position_mismatch'
        candidates = []
        if not reason:
            for line in lines:
                b = line.get('bbox')
                equation = simple_equation(line.get('text'))
                if line.get('page') != page or not equation or not _box(b) or b[2] >= box[0]:
                    continue
                overlap = max(0, min(b[3], box[3])-max(b[1], box[1]))
                if overlap / min(b[3]-b[1], box[3]-box[1]) < .6:
                    continue
                candidates.append((line, equation))
            if len(candidates) != 1:
                reason = 'original_equation_not_unique'
        if not reason:
            line, equation = candidates[0]
            matches = [b for b in facts if b.get('page') == page and b.get('type') == 'formula'
                       and simple_equation(b.get('text')) == equation]
            if len(matches) != 1 or not matches[0].get('id') or labels(matches[0]):
                reason = 'indexed_equation_not_unique'
            elif matches[0].get('indexable') in (False, 0) or matches[0].get('is_generated'):
                reason = 'filtered_formula'
            elif sum(b.get('id') == matches[0]['id'] for b in facts) != 1:
                reason = 'conflicting_formula_identity'
        if reason:
            omissions.append({'label_id': identity, 'page': page, 'reference': number, 'reason': reason})
            continue
        proposals.append({'source': doc, 'page': page, 'reference': number,
                          'formula_id': matches[0]['id'], 'formula_text': matches[0]['text'],
                          'label_id': identity, 'label_text': anchor['text'],
                          'label_bbox': list(box), 'original_formula_text': line['text'],
                          'original_formula_bbox': list(line['bbox']),
                          'method': 'exact_ascii_equation_same_line_v1',
                          'subsection_relation': 'unknown', 'semantic_support_validated': False})
    # The same unnumbered formula must not acquire two different labels.
    conflicts = {p['formula_id'] for p in proposals
                 if sum(q['formula_id'] == p['formula_id'] for q in proposals) != 1}
    for p in proposals:
        if p['formula_id'] in conflicts:
            omissions.append({'label_id': p['label_id'], 'page': p['page'],
                              'reference': p['reference'], 'reason': 'multiple_labels_for_formula'})
    return [p for p in proposals if p['formula_id'] not in conflicts], omissions
