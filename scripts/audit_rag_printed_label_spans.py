"""Read-only exact PDF span audit; compare historical clip rule, never select evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from app.rag.formula_positions import label_position_links, standalone_label, verify_printed_label
from probe_rag_prose_spans import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    import pymupdf
    protocol = {'database_sha256': sha(args.database), 'sources': {
        name: sha(ROOT/name) for name in ['scripts/audit_rag_printed_label_spans.py',
                                          'backend/app/rag/formula_positions.py']},
        'pdfs': {}, 'paid_calls': 0, 'corpus_writes': False, 'defaults_changed': False,
        'scope': 'Whole snapshot numbered-label geometry; not retrieval or semantic validation'}
    checks, omissions = [], []
    db = sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        for row in db.execute('select distinct doc_id from document_blocks order by doc_id'):
            doc = row[0]
            facts = [dict(r) for r in db.execute('select * from document_blocks where doc_id=?', (doc,))]
            for fact in facts:
                fact['bbox'] = json.loads(fact['bbox']) if fact['bbox'] else []
            links, rejected = label_position_links(facts)
            omissions.extend(dict(source=doc, **r) for r in rejected)
            # A printed number must also uniquely identify one formula in the full document.
            from app.rag.formula_links import labels
            counts = Counter(n for f in facts if f['type']=='formula' for n in labels(f))
            paths = list((ROOT/'data/textbooks').glob(f"教材-{doc.split('-')[1]}-*.pdf"))
            if doc.split('-')[1]=='1':
                paths = [p for p in paths if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
            if len(paths)!=1:
                raise ValueError('Cannot uniquely map original PDF: '+doc)
            protocol['pdfs'][doc] = {'path': str(paths[0]), 'sha256': sha(paths[0])}
            with pymupdf.open(paths[0]) as pdf:
                cache = {}
                for link in links:
                    page_index = link['page']-1
                    if not 0<=page_index<len(pdf):
                        checks.append(dict(link, result={'status': 'rejected', 'reason': 'page_out_of_range'}))
                        continue
                    page = pdf[page_index]
                    if page_index not in cache:
                        cache[page_index] = [dict(text=s['text'], bbox=list(s['bbox']))
                            for b in page.get_text('dict')['blocks'] for line in b.get('lines', [])
                            for s in line['spans']]
                    box = pymupdf.Rect(link['label_bbox'])
                    old_text = page.get_text('text', clip=box+(-1,-1,1,1)).strip() if page.rect.contains(box) else ''
                    result = verify_printed_label(link, cache[page_index], list(page.rect))
                    if counts[link['reference']]!=1:
                        result = dict(result, status='rejected', reason='ambiguous_document_number')
                    checks.append(dict(link, legacy_clip_text=old_text,
                        legacy_verified=standalone_label(old_text)==link['reference'], result=result))
            print(doc, len(links), flush=True)
    finally:
        db.close()
    if sha(args.database)!=protocol['database_sha256']:
        raise RuntimeError('Source corpus changed')
    drift = [n for n,digest in protocol['sources'].items() if sha(ROOT/n)!=digest]
    if drift:
        raise RuntimeError('Source implementation drift: '+repr(drift))
    summary = {'linked': len(checks), 'verified': sum(c['result']['status']=='verified' for c in checks),
        'legacy_verified': sum(c.get('legacy_verified', False) for c in checks),
        'newly_verified': sum(c['result']['status']=='verified' and not c.get('legacy_verified') for c in checks),
        'legacy_verified_now_rejected': sum(c.get('legacy_verified', False) and c['result']['status']!='verified' for c in checks),
        'rejections': dict(Counter(c['result'].get('reason') for c in checks if c['result']['status']!='verified')),
        'source_drift': drift, 'corpus_changed': False, 'semantic_support_validated': False,
        'defaults_changed': False, 'paid_calls': 0}
    for name,value in [('protocol', protocol), ('checks', checks), ('omissions', omissions), ('summary', summary)]:
        write(args.out/(name+'.json'), value)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
