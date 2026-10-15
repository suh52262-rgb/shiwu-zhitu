"""Combine a failed reviewer supplement with the untouched valid counterpart.

Reject changed criteria/materials and replacement of an already-valid rating.
This reports engineering evidence, never a formal deployment gate.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from probe_rag_prose_spans import sha, write
from review_position_intro_material import Review, validate_review


def combine(original_protocol, supplemental_protocol, original, supplemental):
    for field in ['input_sha256', 'dataset_sha256', 'rule', 'packets']:
        if original_protocol[field] != supplemental_protocol[field]:
            raise ValueError(f'Review identity mismatch: {field}')
    merged = {}
    for reviewer in ['A', 'B']:
        record = original[reviewer]
        if reviewer in supplemental:
            if record['status'] == 'valid':
                raise ValueError('Cannot replace valid original rating')
            record = supplemental[reviewer]
        if record['status'] != 'valid' or record['reviewer'] != reviewer:
            raise ValueError('Two valid reviewer records required')
        packet = original_protocol['packets'][reviewer]
        if record['mapping'] != packet['mapping']:
            raise ValueError('Reviewer mapping mismatch')
        parsed = Review.model_validate(record['output']).model_dump()
        facts = {arm: {source['id']: source for source in packet['packet'][arm]} for arm in ['X', 'Y']}
        verified = validate_review(parsed, facts)
        merged[reviewer] = {name: verified[arm] for arm, name in record['mapping'].items()}
    disagreements = []
    for name in ['baseline', 'selected']:
        for point in range(2):
            if merged['A'][name]['points'][point]['status'] != merged['B'][name]['points'][point]['status']:
                disagreements.append({'arm': name, 'point': point + 1,
                                      'A': merged['A'][name]['points'][point]['status'],
                                      'B': merged['B'][name]['points'][point]['status']})
    return {'valid_reviews': 2, 'ratings': merged, 'point_disagreements': disagreements,
            'jointly_confirmed_single_case_improvement': all(
                not r['baseline']['sufficient'] and r['selected']['sufficient'] for r in merged.values()),
            'scope': 'Known engineering case, same-family AI reviews; not teacher or deployment acceptance',
            'defaults_changed': False, 'original_failure_preserved': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [args.original / 'protocol.json', args.supplement / 'protocol.json', args.selection / 'result.json']
    protocols = [json.loads(path.read_text(encoding='utf-8')) for path in paths[:2]]
    if sha(paths[2]) != protocols[0]['input_sha256']:
        raise ValueError('Selection no longer matches review input')
    original, supplemental = {}, {}
    for reviewer in ['A', 'B']:
        path = args.original / (reviewer + '.json')
        paths.append(path)
        original[reviewer] = json.loads(path.read_text(encoding='utf-8'))
        path = args.supplement / (reviewer + '.json')
        if path.exists():
            paths.append(path)
            supplemental[reviewer] = json.loads(path.read_text(encoding='utf-8'))
    result = combine(*protocols, original, supplemental)
    result['inputs'] = {str(path.resolve()): sha(path) for path in paths}
    result['script_sha256'] = sha(Path(__file__))
    write(args.out / 'summary.json', result)
    print(json.dumps({'valid_reviews': result['valid_reviews'], 'single_case_improvement': result['jointly_confirmed_single_case_improvement'],
                      'disagreements': result['point_disagreements']}))


if __name__ == '__main__':
    main()
