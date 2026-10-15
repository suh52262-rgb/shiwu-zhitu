"""Focused development regression, not a new holdout or a replacement T4 gate.

Inputs are previously exposed Q016/Q018/Q061. Two fresh same-family AI reviews
receive the writing, question and new hint, never implementation or old ratings.
No keyword-based quality verdict. All raw outputs and actual usage are retained.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
from pydantic import BaseModel
from app.config import get_settings
from app.agent.graph import generate_hint, hint_policy
from app.models.student import StudentState
from app.services.llm.factory import get_llm
from app.services.llm.tracking import track_usage


class Assessment(BaseModel):
    premature_answer_association: bool
    unnecessary_speed_direction: bool
    redundant_required_step: bool
    explanation: str
    quoted_hint: list[str]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory; no overwriting old evidence')
    args.out.mkdir(parents=True)
    settings = get_settings()
    settings.llm_provider = 'deepseek'
    settings.llm_model = settings.tutor_model = 'deepseek-flash'
    settings.llm_timeout = 120
    cases = []
    for qid in ('Q016', 'Q018', 'Q061'):
        packet = json.loads((ROOT/'artifacts/t4_development_v18_20261013/review_packets'/f'{qid}.json').read_text(encoding='utf-8'))
        cases.append({'id': qid, 'statement': packet['statement'],
                      'writing': packet['student_material'][0 if qid == 'Q018' else -1],
                      'complete': qid != 'Q018', 'round_correct': qid != 'Q018', 'starting_hint_level': 0,
                      'request_hint': True})
    rule = ('只评当前提示，不评报告或学生分数。检查三项：'
            '是否把仅已写的中间数直接对应到尚未写出的答案；'
            '题目仅求速率却要求补方向；'
            '充分完成的书写仍被要求重复已有步骤。明确标为可选的巩固练习不算必做重复。'
            'quoted_hint只能引用本提示逐字原文；没有问题允许空引用。不要用关键词自动判。')
    identity = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for base in (ROOT/'backend/app', ROOT/'prompts') for p in sorted(base.rglob('*'))
                if p.suffix in {'.py', '.md'}}
    write(args.out/'protocol.json', {'cases':cases, 'rule':rule, 'source':identity,
        'scope':'exposed development probes; two fresh same-family AI calls; not teacher validation',
        'acceptance_rule':'all three defect flags false in both reviews for all three generated hints; failures remain failures'})
    records = []
    for case in cases:
        state = StudentState(problem_content=case['statement'], current_answer=case['writing'],
                             hint_level=case['starting_hint_level'], is_complete=case['complete'], round_correct=case['round_correct'], request_hint=True)
        state=state.model_copy(update=hint_policy(state))
        with track_usage() as usage:
            generated = generate_hint(state)
        record = {'case':case, 'generated':generated, 'generation_usage':usage.as_dict(), 'reviews':[]}
        write(args.out/f"{case['id']}.json", record)
        hint = generated.get('hint', '')
        for stream in ('A','B'):
            with track_usage() as review_usage:
                outcome = get_llm('tutor').chat_structured_resilient(rule,
                    json.dumps({'question':case['statement'], 'student_writing':case['writing'],
                                'hint_level':state.hint_level, 'hint':hint}, ensure_ascii=False), Assessment)
            value = outcome.value.model_dump() if outcome.value else None
            valid = bool(hint and value is not None and not outcome.degraded
                         and all(quote and quote in hint for quote in value['quoted_hint']))
            if value and any(value[k] for k in ('premature_answer_association','unnecessary_speed_direction','redundant_required_step')):
                valid = valid and bool(value['quoted_hint'])
            record['reviews'].append({'stream':stream, 'value':value, 'valid':valid,
                                      'reasons':outcome.reasons, 'usage':review_usage.as_dict()})
            write(args.out/f"{case['id']}.json", record)
        records.append(record)
        print('COMPLETE '+case['id'], flush=True)
    keys = ('premature_answer_association','unnecessary_speed_direction','redundant_required_step')
    passed = all(r['valid'] and not any(r['value'][key] for key in keys)
                 for record in records for r in record['reviews'])
    drift = [p for p,h in identity.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    write(args.out/'summary.json', {'cases':len(records), 'passed':passed and not drift,
                                   'source_drift':drift, 'scope':'focused development regression only'})


if __name__ == '__main__':
    main()
