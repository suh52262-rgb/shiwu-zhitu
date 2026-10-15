"""Identity-checked AI visual sidecars for isolated range replay only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from app.rag.section_regions import valid_box


def load_reviewed_headers(directory,expected_pdf_hashes):
    root=Path(directory)
    proposal_path=root/'proposals.json'
    proposals=json.loads(proposal_path.read_text(encoding='utf-8'))
    review=json.loads((root/'visual_check.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    if review.get('proposal_sha256')!=digest:raise ValueError('Review/proposal hash mismatch')
    index={}
    for row in review.get('results',[]):
        key=(row.get('source'),row.get('page'),row.get('body_page'))
        if key in index:raise ValueError('Duplicate visual review identity')
        index[key]=row
    if len(index)!=len(proposals):raise ValueError('Incomplete visual review')
    verified=[];seen=set()
    for p in proposals:
        doc=p.get('source');page=p.get('page');body=p.get('body_page')
        key=(doc,page,body)
        if key in seen:raise ValueError('Duplicate proposal identity')
        seen.add(key)
        if (not doc or not isinstance(page,int) or isinstance(page,bool) or page<1
                or not isinstance(body,int) or isinstance(body,bool) or body<1
                or not p.get('text') or not valid_box(p.get('bbox')) or not valid_box(p.get('body_bbox'))):
            raise ValueError('Invalid proposal identity/geometry')
        if not expected_pdf_hashes.get(doc) or protocol.get('pdfs',{}).get(doc,{}).get('sha256')!=expected_pdf_hashes[doc]:
            raise ValueError('Reviewed original PDF identity mismatch')
        r=index.get(key,{})
        if r.get('appears_running_header') is not True or r.get('matching_body_heading_visible') is not True:
            raise ValueError('Header lacks affirmative visual witness')
        verified.append(dict(p,review_proposal_sha256=digest,
                             review_kind='implementation_agent_ai_visual'))
    return verified


def apply_reviewed_headers(headings,sidecars):
    """Match original text AND exact geometry, never exclude an entire page."""
    output=deepcopy(headings);applied=[]
    normalize=lambda s:re.sub(r'\s+','',s or '')
    for h in output:
        matches=[p for p in sidecars if (p['source'],p['page'])==(h.get('source'),h.get('page'))
                 and normalize(p['text'])==normalize(h.get('text'))
                 and valid_box(h.get('bbox'))
                 and max(abs(a-b) for a,b in zip(p['bbox'],h['bbox']))<=.2]
        if len(matches)>1:raise ValueError('Ambiguous reviewed header match')
        if matches:
            h['running_header_verified']=True
            h['running_header_rule']='identity_checked_ai_visual_sidecar_v1'
            h['header_sidecar_sha256']=matches[0]['review_proposal_sha256']
            applied.append({'source':h['source'],'page':h['page'],'text':h['text'],'bbox':h['bbox']})
    return output,applied
