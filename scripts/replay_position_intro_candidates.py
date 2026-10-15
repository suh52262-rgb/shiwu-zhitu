"""Known-query source-layout replay, only on pages already returned/expanded."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.position_intro_candidates import position_intro_candidates
from app.rag.section_regions import valid_box,verify_text_geometry
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--layout',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    import pymupdf
    original=json.loads((a.audit/'protocol.json').read_text(encoding='utf-8'))
    if sha(a.database)!=original['database_sha256']:raise ValueError('Corpus identity mismatch')
    positions=json.loads((a.audit/'checks.json').read_text(encoding='utf-8'))
    witnesses=json.loads(a.layout.read_text(encoding='utf-8'))['witnesses']
    for w in witnesses:
        if original['pdfs'].get(w['source'],{}).get('sha256')!=w['pdf_sha256']:raise ValueError('Layout source identity mismatch')
    frozen=json.loads((a.input/'protocol.json').read_text(encoding='utf-8'))
    ids=frozen['expected_case_ids']
    protocol={'scope':'Known development query replay, reviewed original layout, not independent validation',
        'database_sha256':sha(a.database),'audit_checks_sha256':sha(a.audit/'checks.json'),
        'layout_sha256':sha(a.layout),'sources':{n:sha(ROOT/n) for n in [
            'scripts/replay_position_intro_candidates.py','backend/app/rag/position_intro_candidates.py','backend/app/rag/section_regions.py']},
        'inputs':{},'pdfs':{},'paid_calls':0,'defaults_changed':False,'query_or_target_ids_to_linker':False}
    db=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    pdfs={};results=[]
    try:
        for identity in ids:
            path=a.input/(identity+'.json');protocol['inputs'][str(path)]=sha(path)
            case=json.loads(path.read_text(encoding='utf-8'))
            if case.get('id')!=identity or case.get('status')!='completed':raise ValueError('Incomplete case')
            pool=case['candidate_pool'];pages=set()
            for h in case['hits']:
                pages.add((h['source'],h['page']))
                pages.update((h['source'],b['page']) for b in h.get('context_blocks',[]) if b.get('page') is not None)
            allowed=[w for w in witnesses if (w['source'],w['page']) in pages]
            facts=[dict(r) for doc in sorted({b['source'] for b in pool}) for r in db.execute('select * from document_blocks where doc_id=?',(doc,))]
            for b in facts:b['bbox']=json.loads(b['bbox']) if b['bbox'] else []
            verified=[];omitted=[]
            for w in allowed:
                doc=w['source'];n=w['page'];info=original['pdfs'][doc]
                if doc not in pdfs:
                    if sha(Path(info['path']))!=info['sha256']:raise ValueError('Original PDF changed')
                    pdfs[doc]=pymupdf.open(info['path']);protocol['pdfs'][doc]=info
                page=pdfs[doc][n-1]
                for b in facts:
                    if (b['doc_id'],b['page'],b['type'])!=(doc,n,'text') or not valid_box(b['bbox']):continue
                    witness=verify_text_geometry(b,list(page.rect),lambda box:page.get_text('text',clip=pymupdf.Rect(box)))
                    if witness['status']=='verified':
                        verified.append(b['id']);b['bbox']=witness['bbox']
                    else:omitted.append({'id':b['id'],'reason':witness['reason']})
            used=sum(len(b['text']) for b in pool)
            if len(pool)>48 or used>12000:raise ValueError('Initial pool exceeds budget')
            added,link_omissions=position_intro_candidates(pool,facts,positions,verified,allowed,budget=12000-used,limit=48-len(pool))
            out={'id':identity,'loaded_or_expanded_pages':sorted(pages),'layout_witnesses_used':allowed,
                'added':added,'candidate_pool':pool+added,'link_omissions':link_omissions,
                'text_omissions':omitted,'actual_selection_status':'not_run','semantic_support_validated':False}
            write(a.out/(identity+'.json'),out);results.append(out)
            print(identity,'new blocks',len(added),flush=True)
    finally:
        db.close()
        for pdf in pdfs.values():pdf.close()
    if sha(a.database)!=protocol['database_sha256'] or sha(a.layout)!=protocol['layout_sha256']:raise RuntimeError('Source changed')
    if any(sha(ROOT/n)!=h for n,h in protocol['sources'].items()) or any(sha(Path(n))!=h for n,h in protocol['inputs'].items()):raise RuntimeError('Source/input drift')
    write(a.out/'protocol.json',protocol);write(a.out/'results.json',results)


if __name__=='__main__':main()
