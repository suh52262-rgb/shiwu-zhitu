"""Prepare limited pools for selection; no model, no required-point-based sorting."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from app.rag.region_pool import merge_range_inventory
from probe_rag_prose_spans import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--ranges',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    protocol=json.loads((a.input/'protocol.json').read_text(encoding='utf-8'))
    ids=protocol['expected_case_ids'];rows=[];inputs={}
    sources={n:sha(ROOT/n) for n in ['scripts/prepare_rag_region_materials.py','backend/app/rag/region_pool.py']}
    for identity in ids:
        paths=[a.input/(identity+'.json'),a.ranges/(identity+'.json')]
        for path in paths:inputs[str(path)]=sha(path)
        original,region=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
        if original.get('id')!=identity or region.get('id')!=identity or original.get('status')!='completed':raise ValueError('Incomplete case')
        pool,details=merge_range_inventory(original['candidate_pool'],region['new_range_candidates'])
        byid={b['id']:b for b in pool}
        for b in original['baseline_plan']['sources']:
            source=byid[b['id']]
            if (source['source'],source['page'])!=(b['source'],b['page']) or b['text'] not in source['text']:raise ValueError('Lost fallback source')
        if original['baseline_plan']['content_characters']>1200:raise ValueError('Baseline material over budget')
        row={'id':identity,'query':original['query'],'baseline_plan':original['baseline_plan'],
            'candidate_pool':pool,'assembly':details,'final_material_budget':1200,'selection_status':'not_run'}
        write(a.out/(identity+'.json'),row);rows.append(row)
    if any(sha(Path(n))!=h for n,h in inputs.items()) or any(sha(ROOT/n)!=h for n,h in sources.items()):raise RuntimeError('Input/source drift')
    write(a.out/'protocol.json',{'sources':sources,'inputs':inputs,'expected_case_ids':ids,
        'required_points_to_assembler':False,'paid_calls':0,'defaults_changed':False})
    write(a.out/'results.json',rows)
    print(json.dumps([{'id':r['id'],**r['assembly']} for r in rows],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
