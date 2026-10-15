"""Prospective read-only frozen-corpus prose completion; no paid retrieval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    os.environ.update(RAG_RERANK='0', RAG_QUERY_EXPANSION='0', EMBEDDING_PROVIDER='bge',
                      HF_HUB_OFFLINE='1', RAG_MATERIAL_POLICY='hit_first_total_v1', RAG_CONTEXT_POLICY='legacy')
    # Pin the model to the same verified deployment snapshot rather than relying on current HF cache.
    model_roots = list(args.snapshot.glob('**/config.json'))
    model_roots = [p.parent for p in model_roots if (p.parent/'modules.json').is_file()]
    if len(model_roots) != 1:
        raise ValueError('Cannot locate unique snapshot BGE model')
    os.environ['EMBEDDING_MODEL'] = str(model_roots[0].resolve())
    manifest = json.loads((args.snapshot/'manifest.json').read_text(encoding='utf-8'))
    verified_model = {}
    for p in model_roots[0].rglob('*'):
        if not p.is_file():
            continue
        key = p.relative_to(args.snapshot).as_posix()
        entry = manifest['files'][key]
        digest = sha(p)
        assert digest == entry['sha256'] and p.stat().st_size == entry['bytes']
        verified_model[key] = digest
    from rag_frozen_corpus_adapter import frozen_stores
    engine, adapter, storage = frozen_stores(args.snapshot, args.out)
    from app.rag.block_store import load_blocks
    from app.agent.tools import retrieve_knowledge
    from app.services.rag_selection import catalogue
    from app.rag.prose_spans import prose_companions
    import pymupdf
    dataset = ROOT/'evaluation/datasets/rag_prose_span_probes_20261014.json'
    data = json.loads(dataset.read_text(encoding='utf-8'))
    source_names = ['backend/app/rag/prose_spans.py', 'backend/app/rag/retriever.py',
                    'backend/app/services/rag_selection.py', 'scripts/probe_rag_prose_spans.py',
                    'scripts/rag_frozen_corpus_adapter.py']
    hashes = {n: sha(ROOT/n) for n in source_names}
    write(args.out/'protocol.json', dict(dataset=data, dataset_sha256=sha(dataset),
          sources=hashes, storage=storage, model_files=verified_model, defaults_changed=False))
    rows, corpus, pdfs, pdf_hashes = [], {}, {}, {}
    for case in data['cases']:
        result = dict(id=case['id'], question=case['query'], required_points=case['required_points'], status='started')
        try:
            status = {}; start = time.perf_counter()
            hits = retrieve_knowledge(case['query'], top_k=5, status=status)
            baseline = catalogue(hits)
            result.update(hits=hits, retrieval_status=status, retrieval_sec=time.perf_counter()-start)
            verified, facts = set(), []
            for doc in sorted({s['source'] for s in baseline}):
                if doc not in corpus:
                    corpus[doc] = load_blocks(doc)
                relevant_pages = {s['page'] for s in baseline if s['source'] == doc}
                selected = [b for b in corpus[doc] if b['page'] in relevant_pages]
                facts.extend(selected)
                if doc not in pdfs:
                    prefix = doc.split('-')[1]
                    paths = list((ROOT/'data/textbooks').glob(f'教材-{prefix}-*.pdf'))
                    if prefix == '1':
                        paths = [p for p in paths if ('学习辅导' in p.name if doc.endswith('jieda') else '上册' in p.name)]
                    if len(paths) != 1:
                        raise ValueError('Unknown original PDF '+doc)
                    pdfs[doc] = pymupdf.open(paths[0]); pdf_hashes[doc] = sha(paths[0])
                for block in selected:
                    if block['type'] != 'text' or len(block.get('bbox') or []) != 4:
                        continue
                    page = pdfs[doc][block['page']-1]
                    box = pymupdf.Rect(block['bbox'])
                    if not box.is_valid or not page.rect.contains(box):
                        continue
                    original = page.get_text('text', clip=box+(-.5, -.5, .5, .5))
                    if re.sub(r'\s+', '', original) == re.sub(r'\s+', '', block['text']):
                        verified.add(block['id'])
            candidate, details = prose_companions(baseline, facts, verified)
            byid = {b['id']: b for b in facts}
            for source in candidate:
                b = byid[source['id']]
                assert (b['doc_id'], b['page']) == (source['source'], source['page'])
                assert b['text'].startswith(source['text']) and b['indexable'] and not b['is_generated']
                if source.get('relation'):
                    assert source['id'] in verified and source['text'] == b['text']
            assert details['extra_characters'] <= 1600 and details['changed_blocks'] <= 8
            result.update(status='completed', completion=details, verified_ids=sorted(verified),
                          arms={name: dict(plan=dict(sources=sources, content_characters=sum(len(s['text']) for s in sources)))
                                for name, sources in [('baseline_pool', baseline), ('companion_pool', candidate)]})
        except Exception as error:
            result.update(status='failed', error=type(error).__name__+': '+str(error))
        write(args.out/(case['id']+'.json'), result)
        rows.append(result); print(case['id'], result['status'], result.get('completion', {}).get('extra_characters'), flush=True)
    write(args.out/'results.json', rows)
    write(args.out/'summary.json', dict(completed=sum(r['status']=='completed' for r in rows),
          source_drift=[n for n, digest in hashes.items() if sha(ROOT/n) != digest],
          pdf_hashes=pdf_hashes, paid_calls=0, defaults_changed=False))
    for pdf in pdfs.values():
        pdf.close()
    engine.dispose(); adapter.close()


if __name__ == '__main__':
    main()
