"""Experimental layout proposals, never automatically verified page headers.

PDF text-layer size is an OCR/layout proxy, not a verified printed font size.
Every proposal still needs original-page inspection; no OCR character repair.
"""
import math
import re
from app.rag.section_regions import heading_kind,valid_box


def header_layout_proposals(headings,page_bounds):
    normalize=lambda text:re.sub(r'\s+','',text or '')
    def size(h):
        v=h.get('font_size')
        return v if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 else None
    def margin(h):
        box=h.get('bbox');bounds=page_bounds.get((h.get('source'),h.get('page')))
        if not valid_box(box) or not valid_box(bounds):return None
        if not (bounds[0]<=box[0]<box[2]<=bounds[2] and bounds[1]<=box[1]<box[3]<=bounds[3]):return None
        return box[3]<=bounds[1]+(bounds[3]-bounds[1])*.08
    groups={}
    for h in headings:
        if (h.get('source') and not h.get('contents_page_verified') and h.get('original_verified') and h.get('font_witness_verified')
                and heading_kind(h.get('text'))=='subsection' and size(h)):
            groups.setdefault((h['source'],normalize(h['text'])),[]).append(h)
    proposals=[];omissions=[]
    for (doc,title),group in groups.items():
        top=[h for h in group if margin(h) is True];body=[h for h in group if margin(h) is False]
        if not top:continue
        if len({h['page'] for h in top})<2 or len(body)!=1:
            omissions.append({'source':doc,'title':title,'reason':'missing_or_ambiguous_body_and_repeat'});continue
        witness=body[0]
        for h in top:
            ratio=size(witness)/size(h)
            if ratio<1.25:
                omissions.append({'source':doc,'page':h['page'],'title':title,'reason':'insufficient_text_layer_size_contrast'});continue
            proposals.append({'source':doc,'page':h['page'],'text':h['text'],'bbox':list(h['bbox']),
                'body_page':witness['page'],'body_bbox':list(witness['bbox']),
                'text_layer_size_ratio':ratio,'horizontal_separator_observed':bool(h.get('horizontal_separator_observed')),
                'status':'needs_original_visual_check','running_header_verified':False,
                'method':'same_title_unique_body_size_contrast_v1','semantic_support_validated':False})
    return proposals,omissions
