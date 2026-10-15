"""Compare assembly on saved candidates; lexical coverage is not sufficiency.

Candidates are captured retrieval results, never selected using labelled gold.
The gold is used only after both production plans have been constructed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.services.rag_materials import fragments


def coverage(row, plan):
    text = {}
    for source in plan['sources']:
        text.setdefault(str(source['id']), []).append(source['text'])
    def covered(excerpt):
        return any(excerpt['quote'] in part for part in text.get(str(excerpt['block_id']), []))
    groups = []
    for group in row['evidence_sets']:
        # Recorded coverage objects preserve the original annotated quote.
        quotes = group.get('quotes', [])
        groups.append(bool(quotes) and all(covered(q) for q in quotes))
    return any(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refuse to overwrite a comparison')
    rows, hashes = [], {}
    for file in sorted(args.archive.glob('*.json')):
        row = json.loads(file.read_text(encoding='utf-8'))
        if not row.get('hits') or not row.get('evidence_sets'):
            continue
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        plans = {name: fragments(row['hits'], selected_policy=name)
                 for name in ('legacy', 'hit_first_total_v1')}
        rows.append({'id': row['id'], 'query': row['query'],
                     'required_answer_points': row['required_answer_points'],
                     'plans': plans,
                     'exact_quote_group_coverage': {name: coverage(row, plan) for name, plan in plans.items()},
                     'semantic_sufficiency': 'pending_independent_review'})
    if len(rows) != 11:
        raise ValueError(f'Expected 11 complete archived cases, got {len(rows)}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'kind': 'archived_candidates_development_comparison',
        'paid_requests': 0, 'source_hashes': hashes,
        'implementation_sha256': hashlib.sha256((ROOT/'backend/app/services/rag_materials.py').read_bytes()).hexdigest(),
        'rows': rows,
        'quote_group_coverage': {name: sum(r['exact_quote_group_coverage'][name] for r in rows)
                                 for name in ('legacy', 'hit_first_total_v1')},
        'warning': 'Exact quote coverage does not establish semantic sufficiency or unseen generalisation.'},
        ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
