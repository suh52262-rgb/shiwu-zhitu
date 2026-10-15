"""Full original-PDF heading inventory; mechanical rule comparison, not gold labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.section_regions import heading_kind,mark_running_headers
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False,parents=True)
    import pymupdf
    source=json.loads((args.audit/'protocol.json').read_text(encoding='utf-8'))
    protocol={'source_protocol_sha256':sha(args.audit/'protocol.json'),
        'sources':{name:sha(ROOT/name) for name in ['backend/app/rag/section_regions.py','scripts/audit_rag_heading_safety.py']},
        'pdfs':source['pdfs'],'scope':'All original pages from eight frozen PDFs; no semantic heading gold',
        'comparison':'Historical repeated-top-8-percent v1 vs current production helper v2',
        'paid_calls':0,'defaults_changed':False}
    write(args.out/'protocol.json',protocol)
    rows=[];counts=[];normalize=lambda text:re.sub(r'\s+','',text)
    for doc,info in sorted(source['pdfs'].items()):
        path=Path(info['path'])
        if sha(path)!=info['sha256']:raise ValueError('Original PDF changed: '+doc)
        headings=[];bounds={}
        with pymupdf.open(path) as pdf:
            pages=len(pdf)
            for i,page in enumerate(pdf):
                bounds[(doc,i+1)]=list(page.rect)
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        text=''.join(s['text'] for s in line['spans'])
                        kind=heading_kind(text)
                        if kind or (len(normalize(text))<=40 and re.match(r'\s*\d{1,2}\s*[.．]',text) and re.search(r'[\u4e00-\u9fff]',text)):
                            headings.append({'source':doc,'page':i+1,'bbox':list(line['bbox']),
                                'text':text,'original_verified':bool(kind)})
        legacy={}
        for h in headings:
            if h['bbox'][3]<=bounds[(doc,h['page'])][3]*.08:
                legacy.setdefault(normalize(h['text']),set()).add(h['page'])
        marked=mark_running_headers(headings,bounds)
        for h in marked:
            old=(h['bbox'][3]<=bounds[(doc,h['page'])][3]*.08 and len(legacy.get(normalize(h['text']),set()))>=2)
            rows.append(dict(h,legacy_header=old,header_kind=heading_kind(h['text'])))
        count={'source':doc,'pages':pages,'heading_candidates':len(marked),
            'legacy_headers':sum(h['bbox'][3]<=bounds[(doc,h['page'])][3]*.08 and len(legacy.get(normalize(h['text']),set()))>=2 for h in marked),
            'current_headers':sum(bool(h.get('running_header_verified')) for h in marked)}
        counts.append(count);print(doc,count['pages'],count['heading_candidates'],flush=True)
    if any(sha(ROOT/name)!=digest for name,digest in protocol['sources'].items()):raise RuntimeError('Source drift')
    changes=[r for r in rows if r['legacy_header']!=bool(r.get('running_header_verified'))]
    write(args.out/'headings.json',rows);write(args.out/'changes.json',changes)
    write(args.out/'summary.json',{'documents':counts,'pages':sum(r['pages'] for r in counts),
        'changed_classifications':len(changes),'changed_kinds':dict(Counter(r['header_kind'] or 'opaque' for r in changes)),
        'semantic_gold_labels_available':False,'paid_calls':0,'defaults_changed':False})


if __name__=='__main__':main()
