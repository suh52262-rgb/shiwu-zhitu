"""Prospective queries through actual retrieval/rerank, paired downstream context."""
import argparse,json,sys,hashlib,os,time
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))

CASES=[
 ('sphere_energy','孤立导体球半径为R，带电Q，它的静电能怎样计算？电势的零点怎样选？',['给出导体球静电能关系及半径依赖','说明孤立球电势相对于无穷远的约定']),
 ('wave_positive','简谐波沿x轴正向传播，原点振动y=Acosωt，位置x的振动表达式是什么？为什么有时间延迟？',['给出沿正向传播的相位或时间延迟关系','解释位置x相对于原点延迟x/u']),
 ('spherical_cap','球形电容器内外球半径不同，它的电容怎样计算？不能直接当孤立球的原因是什么？',['给出球形电容器电容与内外半径关系','区分内外两导体与孤立球的边界条件']),
 ('interference_intensity','两束相干光的光强分别为I1和I2，合成光强与相位差有什么关系？什么时候最亮最暗？',['含相位差的光强叠加关系','说明相长相消条件及强度极值']),
 ('monatomic_heat','经典单原子理想气体的摩尔定容热容为什么是3R/2？这个值适用于什么假设？',['说明平动自由度与内能或热容关系','限定经典单原子理想气体条件']),
 ('rod_emf','导体棒在匀强磁场中平动，什么时候感应电动势可以写成Blv？方向如何判定？',['给出Blv适用的棒长速度磁场方向条件','说明正电荷受磁力或等效电势高低的方向判定']),
 ('index_speed_regression','折射率越大，光在介质中是不是走得越快？',['给出n=c/v','说明相速度与折射率成反比']),
 ('isothermal_regression','理想气体等温膨胀内能没变，是不是说明没有吸热？',['等温时理想气体内能不变','由第一定律说明吸热可以用于对外做功'])]

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cases',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=False)
    cases=CASES
    if a.cases:
        supplied=json.loads(a.cases.read_text(encoding='utf-8'))['cases']
        cases=[(c['id'],c['query'],c['required_points']) for c in supplied]
        assert len({c[0] for c in cases})==len(cases) and all(c[1] and len(c[2])==2 for c in cases)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from app.services.rag_materials import fragments,policy
    from app.rag.retriever import _expand_context
    from app.agent.tools import retrieve_knowledge
    from app.rag import block_store
    base_policy=policy();assert os.getenv('RAG_CONTEXT_POLICY','legacy')=='legacy'
    files=['backend/app/rag/formula_links.py','backend/app/rag/retriever.py','backend/app/rag/block_store.py','backend/app/services/rag_materials.py','backend/app/agent/tools.py','backend/app/rag/rerank.py']
    hashes={f:sha(ROOT/f) for f in files}
    write(a.out/'frozen_protocol.json',{'cases':cases,'cases_sha256':sha(a.cases) if a.cases else None,'source_hashes':hashes,'script_sha256':sha(Path(__file__)),
       'baseline_material_policy':base_policy,'rerank':os.getenv('RAG_RERANK','1'),'query_expansion':os.getenv('RAG_QUERY_EXPANSION','0'),
       'scope':'Supplied prospective query probes' if a.cases else '6 prospective query probes and 2 explicitly labelled regressions; same actual retrieval and rerank candidates shared by both arms',
       'activation_rule':'No source/budget/preservation violations; semantic gains, no confirmed regression, and diverse query evidence required. These small probes alone do not authorize a default switch.'})
    rows=[]
    for identity,query,points in cases:
        row={'id':identity,'question':query,'required_points':points,'scope':'regression' if identity.endswith('regression') else 'prospective_probe','status':'started'}
        write(a.out/(identity+'.json'),row)
        status={};start=time.perf_counter()
        try:
            hits=retrieve_knowledge(query,top_k=5,status=status)
            row.update(retrieval_status=status,retrieval_seconds=time.perf_counter()-start,raw_hits=hits,arms={})
            for label,context in [('legacy','legacy'),('formula_refs_v2','formula_refs_v2')]:
                hs=deepcopy(hits)
                _expand_context(hs,context_policy=context)
                plan=fragments(hs,selected_policy=base_policy)
                row['arms'][label]={'hits':hs,'plan':plan}
            old,new=(row['arms'][k]['plan'] for k in ('legacy','formula_refs_v2'))
            assert new['content_characters']<=1200
            assert all(any(s['id']==v['id'] and s['source']==v['source'] and s['page']==v['page'] and v['text'] in s['text'] for s in new['sources']) for v in old['sources'])
            # Validate all actual delivered fragments against the facts database,
            # including formulas not selected as direct hits.
            facts={}
            for plan in [old,new]:
                for s in plan['sources']:
                    key=(s['source'],s['page'])
                    if key not in facts:facts.update(block_store.load_page_blocks(s['source'],[s['page']]))
                    b=next(b for b in facts[key] if b['id']==s['id'])
                    assert s['text'] in b['text'] and b.get('indexable') is not False and not b.get('is_generated')
            row['facts']={doc+':'+str(page):bs for (doc,page),bs in facts.items()}
            row['status']='completed'
        except Exception as exc:row.update(status='failed',error=type(exc).__name__+': '+str(exc),retrieval_status=status)
        write(a.out/(identity+'.json'),row);rows.append(row);print(identity,row['status'],flush=True)
    write(a.out/'results.json',rows)
    write(a.out/'summary.json',{'cases':len(rows),'completed':sum(r['status']=='completed' for r in rows),'source_drift':[f for f,h in hashes.items() if sha(ROOT/f)!=h],'baseline_material_policy':base_policy,'default_changed':False})
if __name__=='__main__':main()
