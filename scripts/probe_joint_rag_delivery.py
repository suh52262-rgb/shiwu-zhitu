"""Prospective actual-material comparison. Selector sees no required-point labels."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from probe_rag_prose_spans import sha, write
from prepare_joint_delivery_review import validate_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
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
            assert manifest['files'][key]['sha256']==digest
            model_hashes[key] = digest
    assert model_hashes and (model/'modules.json').is_file()
    os.environ['EMBEDDING_MODEL'] = str(model.resolve())
    from rag_frozen_corpus_adapter import frozen_stores
    engine, adapter, storage = frozen_stores(args.snapshot, args.out)
    from app.agent.tools import retrieve_knowledge
    from app.rag.retriever import _expand_context
    from app.rag.block_store import load_blocks
    from app.rag.formula_links import labels
    from app.rag.formula_positions import label_position_links, standalone_label
    from app.services.rag_materials import fragments
    from app.services.rag_selection import catalogue
    from app.services.rag_joint_selection import unified_select, JOINT_RULE
    from app.services.llm import get_llm
    import pymupdf
    dataset = ROOT/'evaluation/datasets/rag_joint_delivery_probes_20261014.json'
    cases = json.loads(dataset.read_text(encoding='utf-8'))
    source_names = ['scripts/probe_joint_rag_delivery.py', 'backend/app/services/rag_joint_selection.py',
                    'backend/app/services/rag_selection.py', 'backend/app/services/rag_materials.py',
                    'backend/app/rag/retriever.py', 'backend/app/rag/formula_positions.py']
    hashes = {name: sha(ROOT/name) for name in source_names}
    write(args.out/'protocol.json', dict(dataset=cases, dataset_sha256=sha(dataset),
          sources=hashes, model_hashes=model_hashes, storage=storage, selection_rule=JOINT_RULE,
          required_points_to_selector=False, defaults_changed=False))
    rows, corpus, pdfs, pdf_hashes = [], {}, {}, {}
    budget_blocked = False
    for case in cases['cases']:
        result = dict(id=case['id'], question=case['query'], required_points=case['required_points'], status='started')
        try:
            started = time.perf_counter(); status = {}
            hits = retrieve_knowledge(case['query'], top_k=5, status=status)
            baseline = fragments(hits, selected_policy='hit_first_total_v1')
            baseline_pool = catalogue(hits)
            expanded = deepcopy(hits); _expand_context(expanded, context_policy='formula_refs_v2')
            raw = catalogue(expanded)
            facts = []
            for doc in sorted({s['source'] for s in baseline_pool+raw}):
                if doc not in corpus:
                    corpus[doc] = load_blocks(doc)
                facts.extend(corpus[doc])
            positions, unknown = label_position_links(facts)
            byposition = {p['formula_id']: p for p in positions}
            byid = {b['id']: b for b in facts}
            pool = {s['id']: dict(s) for s in baseline_pool}
            omissions, verified_labels = [], []
            used = sum(len(s['text']) for s in pool.values())
            assert used<=12000 and len(pool)<=48
            for source in raw:
                old = pool.get(source['id'])
                if old and old['text']==source['text']:
                    continue
                b = byid[source['id']]
                if old and not source['text'].startswith(old['text']):
                    omissions.append(dict(id=source['id'], reason='non_extending_prefix')); continue
                if source['type']=='formula' and not old:
                    number = labels(b); position = byposition.get(b['id'])
                    if len(number)!=1 or not position:
                        omissions.append(dict(id=b['id'], reason='unknown_printed_formula_label')); continue
                    matches = [f for f in facts if f['doc_id']==b['doc_id'] and f['type']=='formula' and number.intersection(labels(f))]
                    if len(matches)!=1:
                        omissions.append(dict(id=b['id'], reason='ambiguous_document_number')); continue
                    doc = b['doc_id']
                    if doc not in pdfs:
                        prefix = doc.split('-')[1]
                        paths = list((ROOT/'data/textbooks').glob(f'教材-{prefix}-*.pdf'))
                        if prefix=='1':
                            paths = [p for p in paths if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
                        if len(paths)!=1:
                            raise ValueError('Unknown source PDF')
                        pdfs[doc] = pymupdf.open(paths[0]); pdf_hashes[doc] = sha(paths[0])
                    page = pdfs[doc][position['page']-1]; box = pymupdf.Rect(position['label_bbox'])
                    printed = page.get_text('text', clip=box+(-1,-1,1,1)).strip() if page.rect.contains(box) else ''
                    if standalone_label(printed)!=position['reference']:
                        omissions.append(dict(id=b['id'], reason='original_label_clip_mismatch')); continue
                    verified_labels.append(dict(position, pdf_label_text=printed))
                cost = len(source['text'])-(len(old['text']) if old else 0)
                if used+cost>12000 or (not old and len(pool)>=48):
                    omissions.append(dict(id=source['id'], reason='candidate_pool_budget')); continue
                pool[source['id']] = dict(source); used += cost
            for source in pool.values():
                b = byid[source['id']]
                assert (b['doc_id'],b['page'])==(source['source'],source['page'])
                assert b['text'].startswith(source['text']) and b['indexable'] and not b['is_generated']
            result.update(hits=hits, retrieval_status=status, baseline_pool=baseline_pool,
                          joint_pool=list(pool.values()), omissions=omissions, verified_labels=verified_labels,
                          position_omissions=unknown, candidate_characters=used,
                          retrieval_and_assembly_sec=time.perf_counter()-started)
            if budget_blocked:
                raise RuntimeError('Paid account blocked earlier; no further selector call')
            llm = get_llm('tutor')
            llm.reset_last_call()
            selected = unified_select(list(pool.values()), case['query'], baseline, budget=1200, llm=llm)
            result['selector_call'] = llm.last_call.as_dict() if llm.last_call else None
            result['selection'] = selected
            error = selected.get('selection_error','')
            budget_blocked = '402' in error or 'Insufficient Balance' in error
            validate_material(list(pool.values()), baseline, selected)
            result.update(arms={'baseline_pool': dict(plan=baseline), 'companion_pool': dict(plan=selected)},
                          status='completed' if selected['selection_status']=='ready' else 'failed')
            if selected['selection_status']!='ready':
                result['error'] = selected.get('selection_error', 'Selection was empty')
        except Exception as error:
            result.update(status='failed', error=type(error).__name__+': '+str(error))
        write(args.out/(case['id']+'.json'), result); rows.append(result)
        print(case['id'], result['status'], result.get('selection',{}).get('content_characters'), flush=True)
    write(args.out/'results.json', rows)
    write(args.out/'summary.json', dict(completed=sum(r['status']=='completed' for r in rows),
          source_drift=[name for name,digest in hashes.items() if sha(ROOT/name)!=digest],
          pdf_hashes=pdf_hashes, selector_attempts=sum('selector_call' in r for r in rows),
          paid_account_blocked=budget_blocked, defaults_changed=False))
    for pdf in pdfs.values():
        pdf.close()
    engine.dispose(); adapter.close()


if __name__=='__main__':
    main()
