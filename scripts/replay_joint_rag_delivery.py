"""Replay current production expansion/delivery on archived hits, not a new eval."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from probe_rag_prose_spans import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(exist_ok=False)
    from sqlalchemy import create_engine
    from app.rag import block_store
    from app.rag.retriever import _expand_context
    from app.services.rag_materials import fragments
    from app.services.rag_selection import catalogue
    previous = json.loads((args.input/'protocol.json').read_text(encoding='utf-8'))
    names = ['backend/app/rag/retriever.py', 'backend/app/services/rag_materials.py',
             'backend/app/services/rag_selection.py']
    hashes = {name: sha(ROOT/name) for name in names}
    if any(previous['source_hashes'][name] != digest for name, digest in hashes.items()):
        raise ValueError('Production source changed since recorded run')
    database = args.input/'frozen_corpus.sqlite'
    digest_before = sha(database)
    engine = create_engine('sqlite://', creator=lambda: sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True))
    original = block_store.get_engine; block_store.get_engine = lambda: engine
    records_path = args.input/'results.json'
    records = json.loads(records_path.read_text(encoding='utf-8'))
    assert len(records)==3 and all(r['status']=='completed' for r in records)
    output = []
    for record in records:
        hits = deepcopy(record['hits'])
        _expand_context(hits, context_policy='legacy')
        baseline = fragments(hits, selected_policy='hit_first_total_v1')
        assert baseline == record['actual_delivery']
        _expand_context(hits, context_policy='formula_refs_v2')
        candidate = catalogue(hits)
        delivered = fragments(hits, selected_policy='hit_first_total_v1')
        targets = []
        for item in record['target_diagnoses']:
            identity = item['target']['id']
            pieces = [s for s in delivered['sources'] if s.get('id')==identity]
            assert all((s['source'],s['page']) == (item['source'],item['page']) for s in pieces)
            targets.append(dict(id=identity, source=item['source'], page=item['page'],
                                catalogued=any(s['id']==identity for s in candidate),
                                delivered_whole=any(s['text']==item['text'].strip() for s in pieces),
                                delivered_any=bool(pieces),
                                omissions=[o for o in delivered['omissions'] if o.get('id')==identity]))
        output.append(dict(id=record['id'], baseline=baseline, candidate_delivery=delivered,
                           targets=targets, semantic_support_validated=False))
        print(record['id'], baseline['content_characters'], delivered['content_characters'],
              sum(t['delivered_whole'] for t in targets), flush=True)
    block_store.get_engine = original; engine.dispose()
    assert sha(database)==digest_before
    write(args.out/'protocol.json', dict(input_sha256=sha(records_path), database_sha256=digest_before,
          production_hashes=hashes, script_sha256=sha(Path(__file__)),
          role='Posthoc mechanical delivery replay, not independent semantic acceptance', paid_calls=0))
    write(args.out/'results.json', output)
    write(args.out/'summary.json', dict(cases=len(output), targets=sum(len(r['targets']) for r in output),
          target_catalogued=sum(t['catalogued'] for r in output for t in r['targets']),
          target_delivered_whole=sum(t['delivered_whole'] for r in output for t in r['targets']),
          target_delivered_any=sum(t['delivered_any'] for r in output for t in r['targets']),
          defaults_changed=False, semantic_acceptance='not_evaluated'))


if __name__ == '__main__':
    main()
