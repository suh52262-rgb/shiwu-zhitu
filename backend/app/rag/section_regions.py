"""Experimental original-page subsection ranges, not derivation/paragraph links."""
import math
import re


def heading_kind(text):
    value=re.sub(r'\s+','',text or '').replace('．','.')
    if re.fullmatch(r'\d{1,2}\.\d{1,2}[\u4e00-\u9fff]{2,24}',value):
        return 'subsection'
    if re.fullmatch(r'(?:§\d{1,2}[.、]?|第[一二三四五六七八九十百0-9]+章)[\u4e00-\u9fff]{2,24}',value):
        return 'parent'
    return None


def valid_box(box):
    return (isinstance(box,(list,tuple)) and len(box)==4
        and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
        and box[0]<box[2] and box[1]<box[3])


def mark_running_headers(headings,page_bounds):
    """Conservative repeated margin headers; subsection headings never erased.

    page_bounds maps (source,page) to original page rectangles. Parent headings
    can repeat in margins. OCR-opaque titles additionally need an exact body
    occurrence. This is source layout evidence only, not OCR repair.
    """
    normalize=lambda text:re.sub(r'\s+','',text or '')
    output=[dict(h) for h in headings]
    for h in output:
        h.pop('running_header_verified',None);h.pop('running_header_rule',None)
    def margin(h):
        bounds=page_bounds.get((h.get('source'),h.get('page')))
        return (valid_box(bounds) and valid_box(h.get('bbox'))
                and bounds[0]<=h['bbox'][0]<h['bbox'][2]<=bounds[2]
                and bounds[1]<=h['bbox'][1]<h['bbox'][3]<=bounds[1]+(bounds[3]-bounds[1])*.08)
    for h in output:
        h['margin_position']=margin(h)
        if not margin(h) or heading_kind(h.get('text'))=='subsection':continue
        matching=[other for other in output if other.get('source')==h.get('source')
                  and normalize(other.get('text'))==normalize(h.get('text'))]
        repeated={other['page'] for other in matching if margin(other)}
        body_match=any(not margin(other) and valid_box(other.get('bbox')) for other in matching)
        if len(repeated)>=2 and (heading_kind(h.get('text'))=='parent' or body_match):
            h['running_header_verified']=True
            h['running_header_rule']='repeated_margin_parent_or_body_witness_v2'
    return output


def verify_text_geometry(fact,page_bounds,extract_text):
    """Bound an OCR box only if bounded original PDF text remains exact.

    extract_text receives a page-bounded clip. Only whitespace is ignored; no
    punctuation/symbol repair. Returns sidecar coordinates, never mutates facts.
    """
    box=fact.get('bbox')
    if not valid_box(box) or not valid_box(page_bounds):return {'status':'rejected','reason':'invalid_geometry'}
    bounded=[max(box[0],page_bounds[0]),max(box[1],page_bounds[1]),
             min(box[2],page_bounds[2]),min(box[3],page_bounds[3])]
    if not valid_box(bounded):return {'status':'rejected','reason':'outside_page_geometry'}
    clip=[max(bounded[0]-1,page_bounds[0]),max(bounded[1]-1,page_bounds[1]),
          min(bounded[2]+1,page_bounds[2]),min(bounded[3]+1,page_bounds[3])]
    normalize=lambda value:re.sub(r'\s+','',value)
    if not (fact.get('text') or '').strip() or normalize(extract_text(clip))!=normalize(fact['text']):
        return {'status':'rejected','reason':'original_text_clip_mismatch'}
    return {'status':'verified','bbox':bounded,'stored_bbox':list(box),
        'coordinate_recovery':bounded!=list(box),'stored_text_changed':False,
        'method':'page_intersection_exact_original_text_v1'}


def subsection_region(anchor,headings,*,inspected_pages):
    """A verified formula label must fall between two verified heading barriers.

    Requires complete contiguous original-page inspection across the interval.
    All major/subsection headings close ranges. Repeated subsection names are
    ambiguous. No missing-boundary fallback to page/chapter adjacency.
    """
    rejected={'status':'unknown','semantic_support_validated':False}
    if not anchor.get('source') or not isinstance(anchor.get('page'),int) or not valid_box(anchor.get('label_bbox')):
        return dict(rejected,reason='invalid_anchor')
    source=anchor['source'];page=anchor['page'];box=anchor['label_bbox']
    relevant=[]
    for h in headings:
        if h.get('source')!=source:continue
        if h.get('running_header_verified'):continue
        if not isinstance(h.get('page'),int) or not valid_box(h.get('bbox')):
            return dict(rejected,reason='invalid_heading_geometry')
        relevant.append(h)
    relevant.sort(key=lambda h:(h['page'],h['bbox'][1]))
    for i,left in enumerate(relevant):
        for right in relevant[i+1:]:
            if left['page']==right['page'] and min(left['bbox'][3],right['bbox'][3])>max(left['bbox'][1],right['bbox'][1]):
                return dict(rejected,reason='overlapping_heading_order')
    preceding=[h for h in relevant if (h['page'],h['bbox'][3])<=(page,box[1])]
    following=[h for h in relevant if (h['page'],h['bbox'][1])>=(page,box[3])]
    if not preceding or not following:return dict(rejected,reason='missing_heading_boundary')
    start,end=preceding[-1],following[0]
    if any(not h.get('original_verified') or not heading_kind(h.get('text')) for h in [start,end]):
        return dict(rejected,reason='unverified_heading_barrier')
    if heading_kind(start['text'])!='subsection':return dict(rejected,reason='parent_scope_only')
    if any(h['page']==page and h['bbox'][1]<box[3] and h['bbox'][3]>box[1] for h in relevant):
        return dict(rejected,reason='heading_overlaps_anchor')
    norm=lambda s:re.sub(r'\s+','',s).replace('．','.')
    if sum(norm(h['text'])==norm(start['text']) for h in relevant)!=1:
        return dict(rejected,reason='repeated_heading')
    if start.get('margin_position') is True or end.get('margin_position') is True:
        return dict(rejected,reason='unresolved_margin_boundary')
    if not set(range(start['page'],end['page']+1)).issubset(set(inspected_pages)):
        return dict(rejected,reason='uninspected_page_gap')
    return {'status':'bounded','source':source,'start':dict(start),'end':dict(end),
        'semantic_support_validated':False,'reading_order_validated':False,
        'paragraph_continuity_validated':False}


def region_text_candidates(region,facts,verified_ids,*,excluded_ids=()):
    """Return original, verified text blocks inside the range; captions excluded.

    A range is a candidate search space. Even a full stored block may be a
    sentence fragment or contain a negation. No inferred applicable conditions.
    """
    if region.get('status')!='bounded':return []
    start,end=region['start'],region['end'];verified=set(verified_ids);excluded=set(excluded_ids)
    identities=[b.get('id') for b in facts]
    output=[]
    for b in facts:
        identity=b.get('id');box=b.get('bbox');page=b.get('page')
        if (not identity or identities.count(identity)!=1 or identity not in verified or identity in excluded
                or (b.get('doc_id') or b.get('source'))!=region['source']
                or b.get('type')!='text' or b.get('indexable') in (False,0)
                or b.get('is_generated') or not valid_box(box)
                or not isinstance(page,int) or not re.search(r'[\u4e00-\u9fff]',b.get('text') or '')):continue
        if (page,box[1])<(start['page'],start['bbox'][3]) or (page,box[3])>(end['page'],end['bbox'][1]):continue
        output.append(dict(b,source=region['source'],relation={
            'kind':'original_pdf_subsection_range','start_heading':start['text'],
            'end_heading':end['text'],'semantic_support_validated':False,
            'paragraph_continuity_validated':False,'text_completeness_verified':False}))
    return output
