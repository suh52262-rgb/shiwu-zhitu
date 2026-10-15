"""Production-stage recording on frozen corpus. Posthoc diagnostics, no gate claims."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from probe_rag_prose_spans import sha, write
from rag_stage_attribution import classify


def selected_stage_cases(queries,targets):
    cases=queries.get('cases',[]);ids=[c.get('id') for c in cases]
    if not ids or len(ids)!=len(set(ids)) or any(not c.get('id') or not (c.get('query') or '').strip() for c in cases):
        raise ValueError('Missing/duplicate query identity or query text')
    selected=targets.get('cases',{})
    if not selected or not set(selected).issubset(set(ids)):
        raise ValueError('Missing or unknown target case')
    for items in selected.values():
        identities=[t.get('id') for t in items]
        if not identities or None in identities or len(identities)!=len(set(identities)):
            raise ValueError('Missing or duplicate diagnostic target')
    return [c for c in cases if c['id'] in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queries',type=Path,default=ROOT/'evaluation/datasets/rag_prose_span_probes_20261014.json')
    parser.add_argument('--targets',type=Path,default=ROOT/'evaluation/datasets/rag_prose_missing_point_targets_20261014.json')
    args = parser.parse_args(); args.out.mkdir(exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    os.environ.update(RAG_RERANK='0', RAG_QUERY_EXPANSION='0', EMBEDDING_PROVIDER='bge',
                      HF_HUB_OFFLINE='1', RAG_CONTEXT_POLICY='legacy', RAG_MATERIAL_POLICY='hit_first_total_v1')
    model = args.snapshot/'models/bge-small-zh-v1.5'
    manifest = json.loads((args.snapshot/'manifest.json').read_text(encoding='utf-8'))
    model_hashes = {}
    for p in model.rglob('*'):
        if p.is_file():
            key = p.relative_to(args.snapshot).as_posix(); digest = sha(p)
            assert manifest['files'][key]['sha256'] == digest
            model_hashes[key] = digest
    assert model_hashes and (model/'modules.json').is_file()
    os.environ['EMBEDDING_MODEL'] = str(model.resolve())
    from rag_frozen_corpus_adapter import frozen_stores
    engine, client, storage = frozen_stores(args.snapshot, args.out)
    from app.agent import tools
    from app.rag import block_store
    from app.rag.retriever import _expand_context
    from app.services.rag_selection import catalogue
    from app.services.rag_materials import fragments
    from sqlalchemy import text
    indexed = set()
    with gzip.open(args.snapshot/'deployment/data/qdrant_points.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            indexed.add(str(json.loads(line)['id']))
    queries_path = args.queries
    targets_path = args.targets
    queries = json.loads(queries_path.read_text(encoding='utf-8'))
    targets = json.loads(targets_path.read_text(encoding='utf-8'))
    selected_cases=selected_stage_cases(queries,targets)
    source_names = ['scripts/audit_rag_missing_point_stages.py', 'scripts/rag_stage_attribution.py',
                    'backend/app/agent/tools.py', 'backend/app/rag/retriever.py',
                    'backend/app/services/rag_materials.py', 'backend/app/services/rag_selection.py']
    hashes = {name: sha(ROOT/name) for name in source_names}
    write(args.out/'protocol.json', dict(storage=storage, source_hashes=hashes, model_hashes=model_hashes,
          query_sha256=sha(queries_path), target_sha256=sha(targets_path),
          selected_case_ids=[c['id'] for c in selected_cases], query_count=len(queries['cases']),
          role='Posthoc diagnostics; target IDs never passed to retrieval or candidate assembly',
          overrides=dict(rerank=False, expansion=False, context='legacy'), defaults_changed=False))
    original_query, original_retrieve, original_pages = client.query_points, tools.retrieve, block_store.load_page_blocks
    active = {}; phase = 'retrieval'
    def query_points(*a, **kw):
        result = original_query(*a, **kw)
        active['vector_calls'].append(dict(limit=kw.get('limit'),
            filter=str(kw.get('query_filter')), points=[dict(id=str(p.id), score=p.score) for p in result.points]))
        return result
    def retrieve(*a, **kw):
        result = original_retrieve(*a, **kw)
        active['kept_calls'].append(deepcopy(result))
        return result
    def load_pages(doc, pages):
        result = original_pages(doc, pages)
        active['page_calls'].append(dict(phase=phase, source=doc, pages=list(pages)))
        return result
    client.query_points, tools.retrieve, block_store.load_page_blocks = query_points, retrieve, load_pages
    rows = []
    for case in selected_cases:
        active = dict(id=case['id'], query=case['query'], vector_calls=[], kept_calls=[], page_calls=[], status='started')
        try:
            phase = 'retrieval'; status = {}
            hits = tools.retrieve_knowledge(case['query'], top_k=5, status=status)
            base = catalogue(hits); delivery = fragments(hits, selected_policy='hit_first_total_v1')
            phase = 'returned_context_replay'; replay = deepcopy(hits); _expand_context(replay)
            assert [(h.get('context'), h.get('context_blocks')) for h in hits] == [(h.get('context'), h.get('context_blocks')) for h in replay]
            loaded = {(call['source'], page) for call in active['page_calls'] if call['phase']==phase for page in call['pages']}
            # Existing opt-in formula linking, replayed only as a diagnostic arm.
            # Preserve baseline candidate facts; do not let a changed layout erase them.
            phase = 'explicit_joint_diagnostic'; joint_hits = deepcopy(hits)
            _expand_context(joint_hits, context_policy='formula_refs_v2')
            joint_raw = catalogue(joint_hits); joint = {s['id']: dict(s) for s in base}
            extra, omissions, changed = 0, [], 0
            for s in joint_raw:
                old = joint.get(s['id'])
                if old and not s['text'].startswith(old['text']):
                    if not old['text'].startswith(s['text']):
                        omissions.append(dict(id=s['id'], reason='source_prefix_conflict'))
                    continue
                difference = len(s['text'])-(len(old['text']) if old else 0)
                if difference <= 0:
                    continue
                if extra+difference > 1600 or changed >= 8:
                    omissions.append(dict(id=s['id'], reason='joint_whole_block_budget'))
                    continue
                joint[s['id']] = dict(s); extra += difference; changed += 1
            raw_ids = [p['id'] for call in active['vector_calls'] for p in call['points']]
            kept_ids = [h.get('block_id') for call in active['kept_calls'] for h in call]
            returned_ids = [h.get('block_id') for h in hits]
            diagnoses = []
            with engine.connect() as connection:
                for selection in targets['cases'][case['id']]:
                    records = connection.execute(text('select * from document_blocks where id=:id'), dict(id=selection['id'])).mappings().all()
                    assert len(records)==1
                    b = dict(records[0]); ready = block_store.is_index_visible(block_store.doc_index_status(b['doc_id']))
                    eligible = bool(b['indexable']) and not b['is_generated'] and ready
                    stage = classify(b, indexed_ids=indexed, raw_ids=raw_ids, kept_ids=kept_ids,
                        returned_ids=returned_ids, loaded_pages=loaded, catalogue=base, delivered=delivery['sources'], eligible=eligible)
                    diagnoses.append(dict(target=selection, source=b['doc_id'], page=b['page'],
                        block_index=b['block_index'], text=b['text'], text_sha256=__import__('hashlib').sha256(b['text'].encode()).hexdigest(),
                        indexed=b['id'] in indexed, vector_rank=raw_ids.index(b['id'])+1 if b['id'] in raw_ids else None,
                        kept_rank=kept_ids.index(b['id'])+1 if b['id'] in kept_ids else None,
                        returned_rank=returned_ids.index(b['id'])+1 if b['id'] in returned_ids else None,
                        loaded_for_returned_context=(b['doc_id'],b['page']) in loaded,
                        baseline_catalogued=any(s['id']==b['id'] for s in base),
                        diagnostic_joint_catalogued=b['id'] in joint, attribution=stage))
            active.update(status='completed', hits=hits, retrieval_status=status, baseline_catalogue=base,
                          actual_delivery=delivery, diagnostic_joint_catalogue=list(joint.values()),
                          diagnostic_joint_extra=extra, diagnostic_joint_omissions=omissions,
                          target_diagnoses=diagnoses, semantic_support_validated=False)
        except Exception as error:
            active.update(status='failed', error=type(error).__name__+': '+str(error))
        write(args.out/(case['id']+'.json'), active); rows.append(active)
        print(case['id'], active['status'], [(r['page'],r['attribution']) for r in active.get('target_diagnoses',[])], flush=True)
    client.query_points, tools.retrieve, block_store.load_page_blocks = original_query, original_retrieve, original_pages
    write(args.out/'results.json', rows)
    write(args.out/'summary.json', dict(completed=sum(r['status']=='completed' for r in rows),
          attributions=dict(Counter(d['attribution'] for r in rows for d in r.get('target_diagnoses',[]))),
          source_drift=[name for name, digest in hashes.items() if sha(ROOT/name)!=digest],
          paid_calls=0, defaults_changed=False, semantic_acceptance='not_evaluated'))
    engine.dispose(); client.close()


if __name__ == '__main__':
    main()
