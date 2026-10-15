"""Experimental same-page facts, not inferred formula/paragraph dependencies.

No query/gold input. Chapters are ignored because current imports contain empty
or chapter-wide section values. Bounding boxes order nearby text geometrically;
unlocated formulas remain separate same-page evidence, never joined derivations.
This module is not wired into live retrieval or delivery.
"""
import math,re

def _box(b):
    box=b.get('bbox')
    if not isinstance(box,(list,tuple)) or len(box)!=4:
        return None
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box):
        return None
    return box if box[2]>box[0] and box[3]>box[1] else None

def page_evidence_candidates(pool,hits,facts,*,max_pages=3,text_budget=600,formula_budget=600):
    if min(max_pages,text_budget,formula_budget)<0:
        raise ValueError('Negative page evidence limit')
    existing={s['id']:s for s in pool if s.get('id')}
    byid={};conflicts=set()
    for b in facts:
        if not b.get('id'):continue
        if b['id'] in byid and b!=byid[b['id']]:conflicts.add(b['id'])
        byid[b['id']]=b
    pages={};omissions=[];added=[]
    for rank,h in enumerate(hits,1):
        key=(h.get('source'),h.get('page'))
        if not all(key) or key in pages:continue
        if len(pages)>=max_pages:break
        candidates=[b for b in facts if (b.get('doc_id') or b.get('source'),b.get('page'))==key]
        anchor=next((b for b in candidates if b.get('id')==h.get('block_id')),None)
        if anchor is None:
            omissions.append({'source':key[0],'page':key[1],'reason':'missing_hit_identity'});continue
        pages[key]=rank;box=_box(anchor)
        for kind,budget in [('text',text_budget),('formula',formula_budget)]:
            available=[]
            for b in candidates:
                identity=b.get('id');text=b.get('text') or ''
                if not identity or identity in conflicts or identity in existing or b.get('type')!=kind:continue
                if b.get('indexable') is False or b.get('is_generated') or not text.strip():continue
                if kind=='text' and (len(text.strip())<8 or not _box(b) or not box):continue
                if kind=='formula' and re.fullmatch(r'\s*[（(]?\d+\s*[.．]\s*\d+[）)]?\s*',text):continue
                available.append(b)
            if kind=='text':
                available.sort(key=lambda b:(abs((_box(b)[1]+_box(b)[3]-box[1]-box[3])/2),b['block_index']))
            else:
                available.sort(key=lambda b:b['block_index'])
            used=0
            for b in available:
                if used+len(b['text'])>budget:
                    omissions.append({'id':b['id'],'reason':kind+'_whole_block_budget'});continue
                added.append({**b,'source':key[0],'hit_rank':rank,'direct':False,'truncated':False,
                              'source_start':0,'source_end':len(b['text']),
                              'context_relation':{'kind':'same_page_only','anchor_id':anchor['id'],'derivation_inferred':False}})
                existing[b['id']]=added[-1];used+=len(b['text'])
    return added,omissions
