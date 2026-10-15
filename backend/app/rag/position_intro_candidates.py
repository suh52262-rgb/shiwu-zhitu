"""Experimental original-page layout candidates, never semantic dependencies.

Requires observed single-column layout, exact original prose verification and a
unique verified printed label. No target/query input or block-index adjacency.
"""
import re
from app.rag.formula_links import labels
from app.rag.section_regions import valid_box

INTRO=re.compile(r'(?:为|可写成|表示为|下列条件|下式|如下)[：:]?\s*$')


def position_intro_candidates(pool,facts,positions,verified_text_ids,layout_witnesses,*,budget=1600,limit=8):
    if budget<0 or limit<0:raise ValueError('Negative candidate limit')
    index={};duplicates=set()
    for b in facts:
        if b.get('id') in index:duplicates.add(b['id'])
        elif b.get('id'):index[b['id']]=b
    pool_index={b['id']:b for b in pool}
    if len(pool_index)!=len(pool):raise ValueError('Duplicate baseline candidate identity')
    existing=set(pool_index);verified=set(verified_text_ids)-duplicates
    allowed={(w['source'],w['page']) for w in layout_witnesses if w.get('single_column_body_observed') is True}
    output=[];omissions=[];used=0
    for p in positions:
        f=index.get(p.get('formula_id'));box=p.get('label_bbox');doc=p.get('source');page=p.get('page')
        if (doc,page) not in allowed:continue
        if p.get('result',{}).get('status')!='verified' or not valid_box(box):continue
        number=p.get('reference')
        matches=[b for b in index.values() if b.get('doc_id')==doc and b.get('type')=='formula' and number in labels(b)]
        if (not f or f['id'] in duplicates or len(matches)!=1 or len(labels(f))!=1
                or f.get('page')!=page or f.get('indexable') not in (True,1) or f.get('is_generated')):
            omissions.append({'formula_id':p.get('formula_id'),'reason':'formula_identity_filter_or_number_conflict'});continue
        intros=[]
        for b in index.values():
            bb=b.get('bbox')
            if (b.get('id') not in verified or b.get('doc_id')!=doc or b.get('page')!=page
                    or b.get('type')!='text' or b.get('indexable') not in (True,1) or b.get('is_generated')
                    or not valid_box(bb) or not INTRO.search(b.get('text') or '')):continue
            gap=box[1]-bb[3]
            if 0<=gap<=min(48,3*(bb[3]-bb[1])):intros.append((gap,b))
        intros.sort(key=lambda pair:pair[0])
        if not intros:continue
        if len(intros)>1 and abs(intros[0][0]-intros[1][0])<=1:
            omissions.append({'formula_id':f['id'],'reason':'ambiguous_intro_position'});continue
        intro=intros[0][1];top=intro['bbox'][3]
        barriers=[b for b in index.values() if b.get('doc_id')==doc and b.get('page')==page
                  and b.get('id') not in {intro['id'],p.get('label_id')}
                  and valid_box(b.get('bbox')) and b['bbox'][1]<box[1] and b['bbox'][3]>top
                  and (b.get('type') in {'figure','figure_caption'} or re.search(r'[\u4e00-\u9fff]',b.get('text') or ''))]
        if barriers:
            omissions.append({'formula_id':f['id'],'reason':'intervening_prose_or_figure'});continue
        if any(b['id'] in pool_index and ((pool_index[b['id']].get('source'),pool_index[b['id']].get('page'),pool_index[b['id']].get('text'))
                !=(doc,page,b['text'])) for b in [intro,f]):
            omissions.append({'formula_id':f['id'],'reason':'existing_pair_source_incomplete_or_conflicting'});continue
        group=[b for b in [intro,f] if b['id'] not in existing]
        cost=sum(len(b['text']) for b in group)
        if used+cost>budget or len(output)+len(group)>limit:
            omissions.append({'formula_id':f['id'],'reason':'whole_pair_budget_or_limit'});continue
        relation={'kind':'verified_intro_before_printed_label','intro_id':intro['id'],
            'formula_id':f['id'],'label_id':p['label_id'],'gap_points':intros[0][0],
            'semantic_support_validated':False,'formula_geometry_validated':False,'text_completeness_verified':False}
        for b in group:
            output.append(dict(b,source=doc,truncated=False,source_start=0,source_end=len(b['text']),relation=relation))
            existing.add(b['id'])
        used+=cost
    return output,omissions
