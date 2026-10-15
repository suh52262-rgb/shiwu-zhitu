"""Adjudicate only disputed material sufficiency, preserving both initial streams."""
import json
import sys
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
from pydantic import BaseModel


class Point(BaseModel):
    required_point:str
    covered:bool
    verbatim_material_quotes:list[str]
    reason:str


class Verdict(BaseModel):
    material_sufficient:bool
    points:list[Point]
    conclusion:str


def main(batch,out):
    from app.services.llm import get_llm
    batch=Path(batch);out=Path(out)
    if out.exists():raise ValueError('Use a fresh output file')
    records=[]
    for p in sorted(batch.glob('*.json')):
        record=json.loads(p.read_text(encoding='utf-8'))
        reviews=record.get('reviews',[])
        if len(reviews)!=2 or not all(r['valid'] for r in reviews):continue
        if reviews[0]['value']['material_sufficient']==reviews[1]['value']['material_sufficient']:continue
        packet={'question':record['case']['question'],'required_points':record['case']['points'],
                'actually_included_materials':record['prompt_materials']}
        rule=('逐项审核实际发送的教材片段是否覆盖所有预先定义的required_points。不是审核模型能否凭常识回答，'
              '也不是只审核学生当前错误所在的那个点。每个要点都要逐条原文支持，不能从结论补机制。'
              '不用外部知识或未给的教材。引文只能逐字摘actually_included_materials的text。'
              '若有一个必需要点不覆盖，则material_sufficient必须为false。独立AI裁决，不冒充教师。')
        judge=get_llm('review');outcome=judge.chat_structured_resilient(rule,json.dumps(packet,ensure_ascii=False),Verdict)
        value=outcome.value.model_dump() if outcome.value else None
        source_texts=[source['text'] for source in record['prompt_materials']]
        valid=bool(value and not outcome.degraded and len(value['points'])==len(record['case']['points'])
                   and all(q and any(q in s for s in source_texts) for point in value['points'] for q in point['verbatim_material_quotes'])
                   and all(not point['covered'] or point['verbatim_material_quotes'] for point in value['points'])
                   and value['material_sufficient']==all(point['covered'] for point in value['points']))
        records.append({'case_id':record['case']['id'],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                        'value':value,'valid':valid,'call':judge.last_call.as_dict() if judge.last_call else None})
    out.write_text(json.dumps({'scope':'Fresh AI material sufficiency adjudication; initial ratings unchanged',
                              'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Recorded disputed sufficiency cases',len(records))


if __name__=='__main__':main(*sys.argv[1:])
