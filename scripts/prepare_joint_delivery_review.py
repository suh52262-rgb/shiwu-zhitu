"""Prepare all final plans, including failed-selector fallbacks, without hiding failure."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from probe_rag_prose_spans import sha, write


def validate_material(pool, baseline, selected, budget=1200):
    byid = {s['id']: s for s in pool}
    if len(byid)!=len(pool):
        raise ValueError('Duplicate candidate identity')
    if sum(len(s['text']) for s in selected['sources'])>budget:
        raise ValueError('Material budget exceeded')
    if selected['selection_status']=='failed' and selected['sources']!=baseline['sources']:
        raise ValueError('Failed selector did not retain baseline')
    for source in selected['sources']:
        candidate = byid.get(source.get('id'))
        if not candidate or (source.get('source'),source.get('page'))!=(candidate['source'],candidate['page']):
            raise ValueError('Invalid material provenance')
        if selected['selection_status']=='ready':
            if source['text']!=candidate['text']:
                raise ValueError('Selected candidate text changed')
        elif not source.get('text') or source['text'] not in candidate['text']:
            raise ValueError('Fallback text does not occur in source candidate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(exist_ok=False)
    from app.services.rag_materials import fragments
    original = args.input/'results.json'
    records = json.loads(original.read_text(encoding='utf-8'))
    assert len(records)==4 and len({r['id'] for r in records})==4
    prepared = []
    for record in records:
        selected = deepcopy(record['selection'])
        baseline = fragments(record['hits'], selected_policy='hit_first_total_v1')
        validate_material(record['joint_pool'], baseline, selected)
        prepared.append(dict(id=record['id'], question=record['question'], required_points=record['required_points'],
            status='completed', preparation_status='ready', workflow_status=record['status'],
            original_workflow_error=record.get('error'), selection_status=selected['selection_status'],
            selection_error=selected.get('selection_error'),
            arms={'baseline_pool': dict(plan=baseline), 'companion_pool': dict(plan=selected)}))
    write(args.out/'review_items.json', prepared)
    write(args.out/'protocol.json', dict(original_input_sha256=sha(original), script_sha256=sha(Path(__file__)),
        role='status=completed means review preparation only; original workflow_status preserved',
        failed_workflows=[r['id'] for r in prepared if r['workflow_status']!='completed'],
        production_material_source_sha256=sha(ROOT/'backend/app/services/rag_materials.py'),
        original_records_changed=False))


if __name__=='__main__':
    main()
