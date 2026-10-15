"""Original-line/font witnesses for previously withheld numeric header formats."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.header_layout import header_layout_proposals
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False,parents=True)
    import pymupdf
    original=json.loads((args.inventory/'protocol.json').read_text(encoding='utf-8'))
    changes=json.loads((args.inventory/'changes.json').read_text(encoding='utf-8'))
    normalize=lambda text:re.sub(r'\s+','',text)
    keys={(r['source'],normalize(r['text'])) for r in changes if r['header_kind']=='subsection'}
    rows=[dict(r) for r in json.loads((args.inventory/'headings.json').read_text(encoding='utf-8'))
          if (r['source'],normalize(r['text'])) in keys]
    protocol={'scope':'Known withheld subsection-format header diagnosis, not independent validation',
        'sources':{name:sha(ROOT/name) for name in ['backend/app/rag/header_layout.py','scripts/audit_rag_header_layout.py']},
        'input_hashes':{name:sha(args.inventory/name) for name in ['protocol.json','headings.json','changes.json']},
        'pdfs':original['pdfs'],'criteria':{'repeated_original_pages':2,'unique_exact_body_title':1,'text_layer_size_ratio_min':1.25},
        'paid_calls':0,'defaults_changed':False,'font_size_is_ocr_proxy':True}
    write(args.out/'protocol.json',protocol);bounds={};failures=[]
    for doc in sorted({r['source'] for r in rows}):
        info=original['pdfs'][doc];path=Path(info['path'])
        if sha(path)!=info['sha256']:raise ValueError('PDF changed')
        with pymupdf.open(path) as pdf:
            cache={}
            for r in [r for r in rows if r['source']==doc]:
                n=r['page'];page=pdf[n-1];bounds[(doc,n)]=list(page.rect)
                if n not in cache:
                    cache[n]=[line for b in page.get_text('dict')['blocks'] for line in b.get('lines',[])]
                markers=[line for line in cache[n]
                         if normalize(''.join(s['text'] for s in line['spans'])) in ('目录','目次','CONTENTS')
                         and line['bbox'][3]<=page.rect.height*.2]
                r['contents_page_verified']=bool(markers)
                if markers:r['contents_marker_bboxes']=[list(line['bbox']) for line in markers]
                matches=[line for line in cache[n] if normalize(''.join(s['text'] for s in line['spans']))==normalize(r['text'])
                         and max(abs(a-b) for a,b in zip(line['bbox'],r['bbox']))<=.2]
                if len(matches)!=1:
                    failures.append({'source':doc,'page':n,'text':r['text'],'reason':'original_line_not_unique'});continue
                r['font_size']=max(s['size'] for s in matches[0]['spans']);r['font_witness_verified']=True
                r['horizontal_separator_observed']=False
                for drawing in page.get_drawings():
                    for item in drawing['items']:
                        if item[0]!='l':continue
                        a,b=item[1:3]
                        if abs(a.y-b.y)<=1 and abs(a.x-b.x)>=page.rect.width*.5 and r['bbox'][3]<=a.y<=r['bbox'][3]+30:
                            r['horizontal_separator_observed']=True
        print(doc,'original size witnesses loaded',flush=True)
    proposals,omitted=header_layout_proposals(rows,bounds)
    if any(sha(ROOT/n)!=h for n,h in protocol['sources'].items()):raise RuntimeError('Source drift')
    if any(sha(args.inventory/n)!=h for n,h in protocol['input_hashes'].items()):raise RuntimeError('Input drift')
    for name,value in [('witnesses',rows),('proposals',proposals),('omissions',omitted),('extraction_failures',failures)]:write(args.out/(name+'.json'),value)
    write(args.out/'summary.json',{'input_titles':len(keys),'original_occurrences':len(rows),'proposals':len(proposals),
        'extraction_failures':len(failures),'omission_reasons':dict(Counter(x['reason'] for x in omitted)),
        'vector_separator_proposals':sum(x['horizontal_separator_observed'] for x in proposals),
        'running_headers_automatically_verified':0,'paid_calls':0,'defaults_changed':False})


if __name__=='__main__':main()
