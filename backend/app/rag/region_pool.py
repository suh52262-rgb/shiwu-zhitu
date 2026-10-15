"""Bounded experimental range inventory merge; never queries or evaluation labels."""
from copy import deepcopy


def merge_range_inventory(pool,inventory,*,budget=12000,limit=48):
    if budget<0 or limit<0:raise ValueError('Negative candidate limit')
    output=deepcopy(pool);index={};omissions=[]
    for i,b in enumerate(output):
        if not b.get('id') or b['id'] in index or not b.get('source') or not b.get('page') or not b.get('text'):
            raise ValueError('Invalid baseline candidate identity')
        index[b['id']]=i
    used=sum(len(b['text']) for b in output)
    if len(output)>limit or used>budget:raise ValueError('Baseline exceeds candidate budget')
    added=[];extended=[];seen=set()
    # Stable source/physical position ordering is allocation only, not a claim
    # that all these blocks form a continuous paragraph or logical argument.
    ordered=sorted(inventory,key=lambda b:(b.get('source',''),b.get('page',0),
        b.get('bbox',[0,0,0,0])[1] if len(b.get('bbox') or [])==4 else 0,b.get('id','')))
    for b in ordered:
        identity=b.get('id')
        if (not identity or identity in seen or not b.get('source') or not b.get('page')
                or not b.get('text') or b.get('type')!='text'
                or b.get('indexable') in (False,0) or b.get('is_generated')
                or b.get('relation',{}).get('kind')!='original_pdf_subsection_range'):
            raise ValueError('Invalid range candidate identity/filter/relation')
        seen.add(identity);old=output[index[identity]] if identity in index else None
        if old and ((old['source'],old['page'])!=(b['source'],b['page']) or not b['text'].startswith(old['text'])):
            raise ValueError('Range conflicts with baseline source/text')
        if old and old['text']==b['text']:continue
        cost=len(b['text'])-(len(old['text']) if old else 0)
        if used+cost>budget or (old is None and len(output)>=limit):
            omissions.append({'id':identity,'reason':'whole_candidate_budget_or_limit'});continue
        candidate=dict(deepcopy(b),truncated=False,source_start=0,source_end=len(b['text']),
                       direct=False if old is None else old.get('direct',False))
        if old is None:
            index[identity]=len(output);output.append(candidate);added.append(identity)
        else:output[index[identity]]=candidate;extended.append(identity)
        used+=cost
    return output,{'added_ids':added,'extended_ids':extended,'omissions':omissions,
        'candidate_characters':used,'candidate_count':len(output),
        'semantic_support_validated':False,'actual_delivery_evaluated':False}
