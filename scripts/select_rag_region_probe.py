"""Source-check one frozen engineering probe before an optional paid selection.

Default is offline. --run-selector permits one call, with no retries. This is
not a semantic evaluation, a blind test, or a production-default change.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from probe_rag_prose_spans import sha, write
from prepare_joint_delivery_review import validate_material
from rag_bounded_model_call import SingleAttemptStructuredClient


def check_sources(pool, baseline, connection):
    """Reject fabricated identities, source swaps, rewrites and filtered sources."""
    seen = set()
    records = []
    for candidate in pool:
        identity = candidate.get('id')
        if not identity or identity in seen:
            raise ValueError('Missing or duplicate candidate identity')
        seen.add(identity)
        row = connection.execute(
            'SELECT doc_id,page,block_index,text,indexable,is_generated FROM document_blocks WHERE id=?',
            (identity,),
        ).fetchone()
        if row is None:
            raise ValueError(f'Unknown source identity: {identity}')
        if (candidate.get('source'), candidate.get('page'), candidate.get('block_index')) != tuple(row[:3]):
            raise ValueError(f'Source coordinate mismatch: {identity}')
        if not row[4] or row[5]:
            raise ValueError(f'Filtered source: {identity}')
        text = candidate.get('text')
        if not isinstance(text, str) or not text or text not in row[3]:
            raise ValueError(f'Candidate text not verbatim: {identity}')
        records.append({'id': identity, 'source': row[0], 'page': row[1],
                        'block_index': row[2], 'whole_source': text == row[3],
                        'text_sha256': __import__('hashlib').sha256(text.encode('utf-8')).hexdigest()})
    validate_material(pool, baseline, {'sources': baseline['sources'], 'selection_status': 'failed'})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run-selector', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_path = args.input / (args.case + '.json')
    baseline_path = args.baseline / (args.case + '.json')
    current = json.loads(source_path.read_text(encoding='utf-8'))
    old = json.loads(baseline_path.read_text(encoding='utf-8'))
    cases = json.loads(args.dataset.read_text(encoding='utf-8'))['cases']
    matching = [case for case in cases if case['id'] == args.case]
    if len(matching) != 1 or current['id'] != args.case or old['id'] != args.case or old['query'] != matching[0]['query']:
        raise ValueError('Frozen case/query identity mismatch')
    frozen = {str(path.resolve()): sha(path) for path in [source_path, baseline_path, args.dataset, args.database]}
    sources = {name: sha(ROOT / name) for name in [
        'scripts/select_rag_region_probe.py', 'scripts/prepare_joint_delivery_review.py', 'scripts/rag_bounded_model_call.py',
        'backend/app/services/rag_joint_selection.py', 'backend/app/services/rag_selection.py']}
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        checked = check_sources(current['candidate_pool'], old['baseline_plan'], connection)
    baseline_ids = {block['id'] for block in old['baseline_plan']['sources']}
    protocol = {'scope': 'Known engineering probe, not semantic acceptance or production deployment',
                'case': args.case, 'inputs': frozen, 'sources': sources,
                'source_checks': checked, 'candidate_count': len(checked),
                'new_candidate_ids': [row['id'] for row in checked if row['id'] not in baseline_ids],
                'required_points_to_selector': False, 'defaults_changed': False,
                'max_model_calls': int(args.run_selector), 'max_retries': 0, 'format_retries': 0,
                'timeout_semantics': 'SDK request timeout, not hard wall-clock deadline',
                'timeout_seconds': 60, 'material_budget': 1200}
    write(args.out / 'protocol.json', protocol)
    result = {'id': args.case, 'query': old['query'], 'status': 'source_checks_passed',
              'semantic_sufficiency_evaluated': False, 'model_requests': 0}
    if args.run_selector:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
        os.environ.update(LLM_MAX_RETRIES='0', LLM_TIMEOUT='60')
        from app.services.llm import get_llm
        from app.services.rag_joint_selection import unified_select
        llm = get_llm('tutor')
        if llm.model != 'deepseek-flash':
            raise ValueError('Model differs from authorized model')
        llm = SingleAttemptStructuredClient(llm)
        llm.reset_last_call()
        result['model_requests'] = 1
        try:
            selected = unified_select(current['candidate_pool'], old['query'], old['baseline_plan'], budget=1200, llm=llm)
            validate_material(current['candidate_pool'], old['baseline_plan'], selected)
            result.update(status='completed' if selected['selection_status'] == 'ready' else 'failed',
                          baseline_plan=old['baseline_plan'], selected_plan=selected,
                          model_call=llm.last_call.as_dict() if llm.last_call else None,
                          new_ids_selected=[i for i in selected.get('selected_ids', []) if i not in baseline_ids])
        except Exception as error:
            result.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                          model_call=llm.last_call.as_dict() if llm.last_call else None)
    if any(sha(Path(path)) != value for path, value in frozen.items()) or any(sha(ROOT / path) != value for path, value in sources.items()):
        raise RuntimeError('Frozen input/source drift')
    write(args.out / 'result.json', result)
    print(json.dumps({'case': args.case, 'status': result['status'], 'model_requests': result['model_requests'],
                      'characters': result.get('selected_plan', {}).get('content_characters')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
