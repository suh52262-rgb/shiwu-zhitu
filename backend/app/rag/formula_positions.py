"""Recover printed equation-label positions, never inferred formula boxes/chains."""
import math,re
from app.rag.formula_links import labels
_LABEL=re.compile(r'\s*[（(]\s*(\d+)\s*[.．]\s*(\d+)\s*[）)]\s*')
def standalone_label(text):
    m=_LABEL.fullmatch(text or '')
    return f'{m[1]}.{m[2]}' if m else None

def verify_printed_label(position, spans, page_bounds):
    """Verify a stored label against whole PDF spans, never substring clipping.

    Caller supplies all spans from the identified original page. A unique exact
    standalone label must lie inside the stored anchor (2pt tolerance). No formula
    body, derivation or semantic support is inferred. Split-span labels remain
    unknown rather than joining arbitrary neighbouring text.
    """
    def valid_box(box):
        return (isinstance(box, (list, tuple)) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in box)
                and box[0] < box[2] and box[1] < box[3])

    def contains(outer, inner, tolerance=0):
        return (outer[0]-tolerance <= inner[0] and outer[1]-tolerance <= inner[1]
                and inner[2] <= outer[2]+tolerance and inner[3] <= outer[3]+tolerance)

    result = {'status': 'rejected', 'method': 'whole_pdf_span_label_v1',
              'formula_bbox_known': False, 'semantic_support_validated': False}
    box = position.get('label_bbox')
    reference = position.get('reference')
    if (not valid_box(page_bounds) or not valid_box(box)
            or not contains(page_bounds, box)):
        return dict(result, reason='invalid_anchor_geometry')
    if not reference or standalone_label(position.get('label_text')) != reference:
        return dict(result, reason='stored_label_mismatch')
    matches = [s for s in spans if standalone_label(s.get('text')) == reference]
    if len(matches) != 1:
        return dict(result, reason='original_span_missing_or_ambiguous')
    span = matches[0]
    original_box = span.get('bbox')
    if (not valid_box(original_box) or not contains(page_bounds, original_box)
            or not contains(box, original_box, 2)):
        return dict(result, reason='original_span_position_mismatch')
    return dict(result, status='verified', reason=None,
                original_label_text=span['text'], original_label_bbox=list(original_box))

def label_position_links(blocks):
    groups={};linked=[];omissions=[]
    for b in blocks:
        doc=b.get('doc_id') or b.get('source')
        if doc and b.get('page') is not None:groups.setdefault((doc,b['page']),[]).append(b)
    for (doc,page),facts in groups.items():
        for f in facts:
            if f.get('type')!='formula':continue
            numbers=labels(f)
            if len(numbers)!=1 or not f.get('id'):
                omissions.append({'id':f.get('id'),'reason':'missing_or_conflicting_number'});continue
            n=next(iter(numbers));fs=[b for b in facts if b.get('type')=='formula' and n in labels(b)]
            anchors=[b for b in facts if b.get('type')=='text' and standalone_label(b.get('text'))==n]
            if len(fs)!=1 or len(anchors)!=1:
                omissions.append({'id':f['id'],'reason':'ambiguous_or_missing_label'});continue
            anchor=anchors[0];box=anchor.get('bbox')
            if f.get('indexable') in (False, 0) or anchor.get('indexable') in (False, 0) or f.get('is_generated') or anchor.get('is_generated'):
                omissions.append({'id':f['id'],'reason':'filtered'});continue
            if (not anchor.get('id') or not isinstance(box,(list,tuple)) or len(box)!=4
                    or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
                    or box[2]<=box[0] or box[3]<=box[1]):
                omissions.append({'id':f['id'],'reason':'missing_label_position'});continue
            linked.append({'formula_id':f['id'],'source':doc,'page':page,'reference':n,
                           'label_id':anchor['id'],'label_text':anchor['text'],'label_bbox':list(box),
                           'position_kind':'printed_label_only','formula_bbox_known':False,'derivation_inferred':False})
    return linked,omissions
