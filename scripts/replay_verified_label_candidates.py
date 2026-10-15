"""Replay known development pools using verified span sidecars; no model/gold selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.rag.evidence_companions import evidence_companions
from probe_rag_prose_spans import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(exist_ok=False,parents=True)
    audit=json.loads((args.audit/'protocol.json').read_text(encoding='utf-8'))
    if sha(args.database)!=audit['database_sha256']:
        raise ValueError('Corpus differs from printed-label audit')
    # Verify original PDFs too; a sidecar cannot silently outlive its sources.
    for source in audit['pdfs'].values():
        if sha(Path(source['path']))!=source['sha256']:
            raise ValueError('Original PDF changed')
    checks=json.loads((args.audit/'checks.json').read_text(encoding='utf-8'))
    positions=[dict(c,source_check='verified_label_text') for c in checks if c['result']['status']=='verified']
    input_protocol=args.inputs/'protocol.json'
    frozen=json.loads(input_protocol.read_text(encoding='utf-8'))
    expected=[c['id'] for c in frozen['dataset']['cases']]
    if not expected or len(expected)!=len(set(expected)):
        raise ValueError('Missing or duplicate frozen case identities')
    # Load named frozen cases, not a directory glob that can silently skip failures.
    case_inputs=[]
    for identity in expected:
        path=args.inputs/(identity+'.json')
        old=json.loads(path.read_text(encoding='utf-8'))
        if old.get('id')!=identity or 'joint_pool' not in old:
            raise ValueError('Frozen case missing candidate pool: '+identity)
        case_inputs.append((path,old))
    protocol={'database_sha256':sha(args.database),'audit_sha256':sha(args.audit/'checks.json'),
        'audit_protocol_sha256':sha(args.audit/'protocol.json'),
        'input_protocol_sha256':sha(input_protocol),'expected_case_ids':expected,'inputs':{},
        'sources':{name:sha(ROOT/name) for name in ['backend/app/rag/evidence_companions.py',
                                                'scripts/replay_verified_label_candidates.py']},
        'scope':'Known development replay, candidate mechanics only',
        'required_points_to_assembler':False,'paid_calls':0,'defaults_changed':False,
        'candidate_budget':12000,'candidate_limit':48}
    db=sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[]
    try:
        for path,old in case_inputs:
            protocol['inputs'][str(path)]=sha(path)
            pool=old['joint_pool']
            facts=[dict(r) for source in sorted({b['source'] for b in pool})
                   for r in db.execute('select * from document_blocks where doc_id=? order by page,block_index,id',(source,))]
            for fact in facts:fact['bbox']=json.loads(fact['bbox']) if fact['bbox'] else []
            byid={b['id']:b for b in facts}
            if len(byid)!=len(facts):raise ValueError('Duplicate corpus identity')
            for b in pool:
                stored=byid[b['id']]
                if ((stored['doc_id'],stored['page'],stored['type'])!=(b['source'],b['page'],b['type'])
                        or not stored['text'].startswith(b['text']) or not stored['indexable'] or stored['is_generated']):
                    raise ValueError('Invalid original pool identity/text')
            used=sum(len(b['text']) for b in pool)
            if used>12000 or len(pool)>48:raise ValueError('Original pool outside budget')
            added,omitted=evidence_companions(pool,facts,positions,budget=12000-used,limit=48-len(pool))
            result={'id':old['id'],'baseline_pool':pool,'added':added,'candidate_pool':pool+added,
                'omissions':omitted,'candidate_characters':used+sum(len(b['text']) for b in added),
                'actual_material_delivery_evaluated':False,'semantic_sufficiency_evaluated':False}
            write(args.out/(old['id']+'.json'),result);rows.append(result)
            print(old['id'], 'new candidates',len(added),flush=True)
    finally:db.close()
    if sha(args.database)!=protocol['database_sha256']:raise RuntimeError('Corpus changed')
    if any(sha(ROOT/name)!=digest for name,digest in protocol['sources'].items()):raise RuntimeError('Source drift')
    if any(sha(Path(name))!=digest for name,digest in protocol['inputs'].items()):raise RuntimeError('Input drift')
    if sha(input_protocol)!=protocol['input_protocol_sha256']:raise RuntimeError('Protocol drift')
    if [r['id'] for r in rows]!=expected:raise RuntimeError('Incomplete replay')
    write(args.out/'protocol.json',protocol);write(args.out/'results.json',rows)
    write(args.out/'summary.json',{'cases':len(rows),'new_candidates':sum(len(r['added']) for r in rows),
        'omission_reasons':dict(Counter(o['reason'] for r in rows for o in r['omissions'])),
        'actual_material_delivery_evaluated':False,'semantic_sufficiency_evaluated':False,
        'corpus_changed':False,'paid_calls':0,'defaults_changed':False})


if __name__=='__main__':main()
