"""Read-only original-PDF audit; proposals are sidecars, never database patches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.rag.formula_metadata import recover_missing_numbers
from app.rag.formula_positions import standalone_label


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    import pymupdf
    db = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    protocol = dict(database_sha256=sha(args.database), sources={}, pdfs={},
                    writes_to_corpus=False, scope='printed-number recovery; no semantic or subsection validation')
    for name in ['scripts/audit_formula_metadata_recovery.py', 'backend/app/rag/formula_metadata.py']:
        protocol['sources'][name] = sha(ROOT / name)
    proposals, omissions, counts = [], [], []
    for row in db.execute('select distinct doc_id from document_blocks order by doc_id'):
        doc = row[0]
        prefix = doc.split('-')[1]
        paths = list((ROOT / 'data/textbooks').glob(f'教材-{prefix}-*.pdf'))
        if prefix == '1':
            paths = [p for p in paths if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
        if len(paths) != 1:
            raise ValueError(f'Cannot identify unique source PDF: {doc}')
        path = paths[0]
        protocol['pdfs'][doc] = dict(path=str(path), sha256=sha(path))
        facts = [dict(r) for r in db.execute('select * from document_blocks where doc_id=? order by page,block_index,id', (doc,))]
        for b in facts:
            b['bbox'] = json.loads(b['bbox']) if b['bbox'] else []
        pages = {b['page'] for b in facts if b['type'] == 'text' and standalone_label(b['text'])}
        lines = []
        with pymupdf.open(path) as pdf:
            for page in sorted(pages):
                for block in pdf[page-1].get_text('dict')['blocks']:
                    for line in block.get('lines', []):
                        lines.append(dict(page=page, bbox=list(line['bbox']),
                                          text=''.join(s['text'] for s in line['spans'])))
        recovered, rejected = recover_missing_numbers(facts, lines)
        proposals.extend(recovered)
        omissions.extend(dict(source=doc, **r) for r in rejected)
        counts.append(dict(source=doc, proposals=len(recovered), omitted=len(rejected),
                           inspected_pages=len(pages), indexed_formulas=sum(b['type']=='formula' for b in facts),
                           formulas_without_position=sum(b['type']=='formula' and not b['bbox'] for b in facts),
                           empty_section=sum(not b['section'] for b in facts), blocks=len(facts)))
        print(json.dumps(counts[-1], ensure_ascii=False), flush=True)
    db.close()
    if sha(args.database) != protocol['database_sha256']:
        raise RuntimeError('Source database changed during audit')
    summary = dict(proposals=len(proposals), omissions=len(omissions),
                   omission_reasons=dict(Counter(r['reason'] for r in omissions)), documents=counts,
                   default_changed=False, corpus_changed=False, semantic_support_validated=False)
    for name, value in [('protocol', protocol), ('proposals', proposals), ('omissions', omissions), ('summary', summary)]:
        (args.out / (name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
