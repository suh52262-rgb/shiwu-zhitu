"""Read-only, posthoc source inventory; never a sufficiency or retrieval score.

Search both frozen OCR blocks and original PDF text layers. Whitespace is
removed only for matching; recorded source text is unchanged. Page cooccurrence
is a candidate signal, not proof that two terms share a physical context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pymupdf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signals(text: str) -> list[str]:
    compact = re.sub(r'\s+', '', text)
    string = bool(re.search('弦|绳', compact))
    wave = bool(re.search('波速|横波|驻波|波动|线密度', compact))
    uniform = bool(re.search('均匀|均质|均一|密度不变|密度恒定|密度相同', compact))
    result = []
    if string and wave:
        result.append('string_wave_or_density')
    if uniform and string:
        result.append('uniform_string_cooccurrence')
    if uniform and wave:
        result.append('uniform_wave_cooccurrence')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--pdf-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.pdf_protocol.read_text(encoding='utf-8'))['pdfs']
    verified = {}
    for doc, item in mapping.items():
        path = Path(item['path'])
        actual = sha256(path)
        if actual != item['sha256']:
            raise ValueError(f'Original PDF changed: {doc}')
        verified[doc] = {'path': str(path.resolve()), 'sha256': actual}
    connection = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = []
        for record in connection.execute('SELECT id,doc_id,page,block_index,type,text,indexable,is_generated FROM document_blocks'):
            item = dict(record)
            found = signals(item['text'] or '')
            if found:
                item['signals'] = found
                rows.append(item)
        page_facts = []
        for doc, item in verified.items():
            with pymupdf.open(item['path']) as pdf:
                for offset, page in enumerate(pdf):
                    text = page.get_text()
                    found = signals(text)
                    if found:
                        page_facts.append({'doc_id': doc, 'page': offset + 1,
                                           'signals': found, 'text': text})
    finally:
        connection.close()
    output = {'scope': 'Posthoc lexical inventory, not sufficient-condition annotation or retrieval evaluation',
              'database_sha256': sha256(args.database),
              'script_sha256': sha256(Path(__file__)),
              'pdf_protocol_sha256': sha256(args.pdf_protocol),
              'pdfs': verified, 'block_count': len(rows), 'page_count': len(page_facts),
              'blocks': rows, 'pages': page_facts}
    (args.output / 'inventory.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'blocks': len(rows), 'pages': len(page_facts),
                      'uniform_pages': [{k: x[k] for k in ('doc_id', 'page', 'signals')}
                                        for x in page_facts if any('uniform' in s for s in x['signals'])]}, ensure_ascii=True))


if __name__ == '__main__':
    main()
