"""Experimental evidence companions from explicit equation references only.

No query/gold input; no geometric or subsection dependency inference. A unique
same-document equation must have an independently verified printed-label link.
The output is extra candidates, never an automatic student answer or delivery.
"""
import re
from app.rag.formula_links import labels
_REF=re.compile(r'(?:式\s*[（(]\s*(\d+)\s*[.．]\s*(\d+)\s*[）)]|[（(]\s*(\d+)\s*[.．]\s*(\d+)\s*[）)]\s*式)')
def equation_references(text):
    return list(dict.fromkeys(f'{a or c}.{b or d}' for a,b,c,d in _REF.findall(text or '')))

def evidence_companions(pool,facts,positions,*,budget=1600,limit=8):
    if budget<0 or limit<0:raise ValueError('Negative companion limit')
    index={};conflicts=set()
    for b in facts:
        identity=b.get('id')
        if not identity:continue
        if identity in index and index[identity]!=b:conflicts.add(identity)
        index[identity]=b
    verified={p['formula_id']:p for p in positions if p.get('source_check')=='verified_label_text'}
    existing={s['id'] for s in pool if s.get('id')};requests={};omissions=[]
    for a in pool:
        doc=a.get('source');identity=a.get('id')
        if not doc or not identity:continue
        stored=index.get(identity)
        if (not stored or identity in conflicts
                or (stored.get('doc_id') or stored.get('source'))!=doc
                or stored.get('page')!=a.get('page')
                or not (stored.get('text') or '').startswith(a.get('text') or '')
                or stored.get('indexable') in (False,0) or stored.get('is_generated')):
            omissions.append({'id':identity,'reason':'invalid_or_filtered_anchor'});continue
        nums=equation_references(a.get('text')) if a.get('type')=='text' else list(labels(a)) if a.get('type')=='formula' else []
        for n in nums:requests.setdefault((doc,n),[]).append(identity)
    added=[];used=0
    for (doc,n),anchors in requests.items():
        matches=[b for b in index.values() if (b.get('doc_id') or b.get('source'))==doc and b.get('type')=='formula' and n in labels(b)]
        if len(matches)!=1 or len(labels(matches[0]))!=1:
            omissions.append({'source':doc,'reference':n,'reason':'ambiguous_or_missing_formula'});continue
        f=matches[0];pos=verified.get(f['id'])
        if not pos or (pos['source'],pos['page'],pos['reference'])!=(doc,f['page'],n):
            omissions.append({'source':doc,'reference':n,'reason':'printed_label_unverified'});continue
        if f['id'] in conflicts or f.get('indexable') in (False,0) or f.get('is_generated'):
            omissions.append({'source':doc,'reference':n,'reason':'formula_filtered_or_conflicting'});continue
        candidates=[f]+[b for b in index.values() if (b.get('doc_id') or b.get('source'))==doc and b.get('type')=='text' and n in equation_references(b.get('text'))]
        for b in candidates:
            if b['id'] in existing or b['id'] in conflicts or b.get('indexable') in (False,0) or b.get('is_generated') or not (b.get('text') or '').strip():continue
            if len(added)>=limit or used+len(b['text'])>budget:
                omissions.append({'id':b['id'],'reference':n,'reason':'whole_block_budget_or_limit'});continue
            added.append({**b,'source':doc,'truncated':False,'source_start':0,'source_end':len(b['text']),
                'relation':{'kind':'explicit_equation_reference','reference':n,'formula_id':f['id'],'label_id':pos['label_id'],'anchor_ids':list(dict.fromkeys(anchors)),
                            'semantic_support_validated':False,'subsection_relation':'unknown'}})
            used+=len(b['text']);existing.add(b['id'])
    return added,omissions
