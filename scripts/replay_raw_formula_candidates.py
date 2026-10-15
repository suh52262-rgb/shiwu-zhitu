"""Offline comparison of recorded raw top-30 formulas; no gold or model calls."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.rag.raw_formula_candidates import raw_formula_candidates
from probe_rag_prose_spans import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = {str(path.resolve()): sha(path) for path in [args.recording, args.pool, args.database]}
    sources = {name: sha(ROOT / name) for name in [
        'scripts/replay_raw_formula_candidates.py', 'backend/app/rag/raw_formula_candidates.py']}
    record = json.loads(args.recording.read_text(encoding='utf-8'))
    current = json.loads(args.pool.read_text(encoding='utf-8'))
    if record['id'] != current['id'] or record['status'] != 'completed' or len(record['vector_calls']) != 1:
        raise ValueError('Incomplete or ambiguous recording')
    raw = record['vector_calls'][0]
    if raw['limit'] != 30 or len(raw['points']) != 30:
        raise ValueError('Expected exact recorded top-30')
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        facts = [dict(row) for point in raw['points'] for row in connection.execute(
            'SELECT * FROM document_blocks WHERE id=?', (point['id'],))]
    output, metadata = raw_formula_candidates(raw['points'], current['candidate_pool'], facts)
    result = {'id': record['id'], 'candidate_pool': output, 'preservation': metadata,
              'selection_status': 'not_run', 'raw_and_retained_distinguished': True}
    protocol = {'inputs': frozen, 'sources': sources, 'scope': 'Known development top-30 recording replay',
                'gold_ids_to_assembler': False, 'query_to_assembler': False,
                'max_added': 2, 'paid_calls': 0, 'defaults_changed': False}
    if any(sha(Path(path)) != value for path, value in frozen.items()) or any(sha(ROOT / path) != value for path, value in sources.items()):
        raise RuntimeError('Frozen source drift')
    write(args.out / 'result.json', result)
    write(args.out / (record['id'] + '.json'), result)
    write(args.out / 'protocol.json', protocol)
    print(json.dumps(metadata, ensure_ascii=True))


if __name__ == '__main__':
    main()
