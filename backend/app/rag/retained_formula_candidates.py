"""Experimental preservation of already-retained vector formulas, not new search."""
from copy import deepcopy


def retained_formula_candidates(retained,pool,facts,*,budget=12000,limit=48,max_added=2):
    if min(budget,limit,max_added)<0:raise ValueError('Negative pool limit')
    out=deepcopy(pool);existing={b['id'] for b in out};byid={b['id']:b for b in facts}
    if len(existing)!=len(out) or len(byid)!=len(facts):raise ValueError('Duplicate identity')
    used=sum(len(b['text']) for b in out)
    if used>budget or len(out)>limit:raise ValueError('Baseline over budget')
    added=[];omissions=[]
    for rank,h in enumerate(retained,1):
        if h.get('type')!='formula':continue
        identity=h.get('block_id')
        if not identity:
            omissions.append({'id':None,'reason':'unverifiable_retained_identity'});continue
        if identity in existing:continue
        b=byid.get(identity)
        if (not b or b.get('type')!='formula' or b.get('indexable') in (False,0) or b.get('is_generated')
                or (b.get('doc_id'),b.get('page'),b.get('text'))!=(h.get('source'),h.get('page'),h.get('text'))):
            omissions.append({'id':identity,'reason':'retained_source_text_or_filter_mismatch'});continue
        if len(added)>=max_added or len(out)>=limit or used+len(b['text'])>budget:
            omissions.append({'id':identity,'reason':'whole_candidate_budget_or_limit'});continue
        source=dict(deepcopy(b),source=b['doc_id'],source_start=0,source_end=len(b['text']),truncated=False,
            relation={'kind':'already_retained_vector_formula','retained_rank':rank,
                      'semantic_support_validated':False,'formula_geometry_validated':False})
        out.append(source);existing.add(identity);added.append(identity);used+=len(b['text'])
    return out,{'added_ids':added,'omissions':omissions,'candidate_characters':used,
        'actual_delivery_evaluated':False,'semantic_support_validated':False}
