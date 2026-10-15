"""Use recorded production kept calls; no target IDs or new retrieval to assembler."""
import argparse,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.retained_formula_candidates import retained_formula_candidates
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    input_path=a.input/'results.json';rows=json.loads(input_path.read_text(encoding='utf-8'))
    protocol={'input_sha256':sha(input_path),'database_sha256':sha(a.database),
        'sources':{n:sha(ROOT/n) for n in ['scripts/replay_retained_formula_candidates.py','backend/app/rag/retained_formula_candidates.py']},
        'scope':'Known two-query development replay of original retained formulas','max_added':2,
        'target_ids_to_assembler':False,'paid_calls':0,'defaults_changed':False}
    db=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;results=[]
    try:
        for row in rows:
            if row['status']!='completed' or len(row['kept_calls'])!=1:raise ValueError('Missing/ambiguous production kept call')
            kept=row['kept_calls'][0];pool=row['baseline_catalogue']
            facts=[dict(r) for doc in sorted({h['source'] for h in kept}) for r in db.execute('select * from document_blocks where doc_id=?',(doc,))]
            out,meta=retained_formula_candidates(kept,pool,facts)
            result={'id':row['id'],'candidate_pool':out,'preservation':meta,'selection_status':'not_run'}
            results.append(result);write(a.out/(row['id']+'.json'),result)
    finally:db.close()
    if sha(input_path)!=protocol['input_sha256'] or sha(a.database)!=protocol['database_sha256'] or any(sha(ROOT/n)!=h for n,h in protocol['sources'].items()):raise RuntimeError('Source drift')
    write(a.out/'protocol.json',protocol);write(a.out/'results.json',results)
    print(json.dumps([{'id':r['id'],**r['preservation']} for r in results],ensure_ascii=False))


if __name__=='__main__':main()
