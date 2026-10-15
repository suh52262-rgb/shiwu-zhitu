"""Original-PDF subsection inventory on known pools; no automatic delivery or model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.section_regions import heading_kind,subsection_region,region_text_candidates,valid_box,verify_text_geometry,mark_running_headers
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--header-sidecar',type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    import pymupdf
    original=json.loads((a.audit/'protocol.json').read_text(encoding='utf-8'))
    if sha(a.database)!=original['database_sha256']:raise ValueError('Corpus mismatch')
    checks=json.loads((a.audit/'checks.json').read_text(encoding='utf-8'))
    positions={c['formula_id']:c for c in checks if c['result']['status']=='verified'}
    input_protocol=json.loads((a.inputs/'protocol.json').read_text(encoding='utf-8'))
    caseids=input_protocol['expected_case_ids']
    if not caseids or len(caseids)!=len(set(caseids)):raise ValueError('Invalid case identities')
    protocol={'database_sha256':sha(a.database),'inputs':{},'pdfs':{},
        'audit_protocol_sha256':sha(a.audit/'protocol.json'),'checks_sha256':sha(a.audit/'checks.json'),
        'sources':{name:sha(ROOT/name) for name in ['backend/app/rag/section_regions.py',
                                                 'scripts/audit_rag_subsection_regions.py']},
        'window_pages_each_side':2,'scope':'Known development range inventory, not delivery or independent quality test',
        'required_points_to_inventory':False,'paid_calls':0,'defaults_changed':False}
    sidecars=[]
    if a.header_sidecar:
        from app.rag.reviewed_headers import load_reviewed_headers,apply_reviewed_headers
        sidecars=load_reviewed_headers(a.header_sidecar,{doc:p['sha256'] for doc,p in original['pdfs'].items()})
        protocol['header_sidecar_files']={name:sha(a.header_sidecar/name) for name in ['protocol.json','proposals.json','visual_check.json']}
        protocol['sources']['backend/app/rag/reviewed_headers.py']=sha(ROOT/'backend/app/rag/reviewed_headers.py')
    db=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    docs={};facts={};results=[]
    normalize=lambda value:re.sub(r'\s+','',value)
    try:
        for identity in caseids:
            path=a.inputs/(identity+'.json');protocol['inputs'][str(path)]=sha(path)
            old=json.loads(path.read_text(encoding='utf-8'))
            if old.get('id')!=identity or 'candidate_pool' not in old:raise ValueError('Incomplete case')
            pool=old['candidate_pool'];existing={b['id'] for b in pool};records=[];potential={}
            for source in pool:
                if source['type']!='formula':continue
                pos=positions.get(source['id'])
                if not pos:records.append({'formula_id':source['id'],'region':{'status':'unknown','reason':'unverified_label'}});continue
                doc=source['source']
                if (doc,source['page'])!=(pos['source'],pos['page']):raise ValueError('Formula source mismatch')
                if doc not in docs:
                    pdfinfo=original['pdfs'][doc];pdfpath=Path(pdfinfo['path'])
                    if sha(pdfpath)!=pdfinfo['sha256']:raise ValueError('Original PDF changed')
                    docs[doc]=pymupdf.open(pdfpath);protocol['pdfs'][doc]=pdfinfo
                    facts[doc]=[dict(r) for r in db.execute('select * from document_blocks where doc_id=?',(doc,))]
                    for b in facts[doc]:b['bbox']=json.loads(b['bbox']) if b['bbox'] else []
                pdf=docs[doc];pages=list(range(max(1,pos['page']-2),min(len(pdf),pos['page']+2)+1))
                headings=[]
                for num in pages:
                    for block in pdf[num-1].get_text('dict')['blocks']:
                        for line in block.get('lines',[]):
                            text=''.join(s['text'] for s in line['spans'])
                            if heading_kind(text):
                                headings.append({'source':doc,'page':num,'bbox':list(line['bbox']),
                                                 'text':text,'original_verified':True})
                            elif (len(normalize(text))<=40 and re.match(r'\s*\d{1,2}\s*[.．]',text)
                                  and re.search(r'[\u4e00-\u9fff]',text)):
                                headings.append({'source':doc,'page':num,'bbox':list(line['bbox']),
                                                 'text':text,'original_verified':False})
                headings=mark_running_headers(headings,{(doc,num):list(pdf[num-1].rect) for num in pages})
                applied=[]
                if sidecars:headings,applied=apply_reviewed_headers(headings,sidecars)
                region=subsection_region(pos,headings,inspected_pages=pages)
                verified=[];rejected=[];excluded=[];recoveries=[]
                nearby=[dict(b) for b in facts[doc] if b['page'] in pages]
                for b in nearby:
                    if b['type']!='text' or not valid_box(b['bbox']):continue
                    page=pdf[b['page']-1]
                    witness=verify_text_geometry(b,list(page.rect),lambda clip:page.get_text('text',clip=pymupdf.Rect(clip)))
                    if witness['status']=='verified':
                        verified.append(b['id'])
                        if witness['coordinate_recovery']:
                            b['bbox']=witness['bbox'];b['stored_bbox']=witness['stored_bbox']
                            recoveries.append(dict(id=b['id'],**witness))
                        if any(h.get('running_header_verified') and h['page']==b['page']
                               and normalize(h['text'])==normalize(b['text']) for h in headings):excluded.append(b['id'])
                    else:rejected.append({'id':b['id'],'reason':witness['reason']})
                candidates=region_text_candidates(region,nearby,verified,excluded_ids=excluded)
                for b in candidates:
                    if b['id'] not in existing:potential[b['id']]=b
                records.append({'formula_id':source['id'],'region':region,'inspected_pages':pages,
                    'headings':headings,'range_blocks':candidates,'text_verification_omissions':rejected,
                    'geometry_sidecars':recoveries,'excluded_running_header_ids':excluded})
                records[-1]['reviewed_headers_applied']=applied
            result={'id':identity,'regions':records,'new_range_candidates':list(potential.values()),
                'inventory_characters':sum(len(b['text']) for b in potential.values()),
                'within_delivery_budget_evaluated':False,'semantic_sufficiency_evaluated':False}
            write(a.out/(identity+'.json'),result);results.append(result)
            print(identity,'potential text blocks',len(potential),flush=True)
    finally:
        db.close()
        for pdf in docs.values():pdf.close()
    if sha(a.database)!=protocol['database_sha256']:raise RuntimeError('Corpus changed')
    if a.header_sidecar and any(sha(a.header_sidecar/n)!=h for n,h in protocol['header_sidecar_files'].items()):
        raise RuntimeError('Reviewed sidecar drift')
    for group in ['sources','inputs']:
        if any(sha(ROOT/name if group=='sources' else Path(name))!=digest for name,digest in protocol[group].items()):
            raise RuntimeError('Source/input drift')
    write(a.out/'protocol.json',protocol);write(a.out/'results.json',results)
    regions=[x['region'] for row in results for x in row['regions']]
    write(a.out/'summary.json',{'cases':len(results),'formula_regions':len(regions),
        'bounded_regions':sum(r['status']=='bounded' for r in regions),
        'unknown_reasons':dict(Counter(r['reason'] for r in regions if r['status']=='unknown')),
        'new_range_candidates':sum(len(r['new_range_candidates']) for r in results),
        'paid_calls':0,'defaults_changed':False,'semantic_sufficiency_evaluated':False})


if __name__=='__main__':main()
