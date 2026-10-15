"""Mechanical enum extraction, keeping the original AI judgements immutable.

Only point-status strings such as '1: covered' or '要点…：missing' are converted.
No boolean, quotation, explanation, material or ordering is edited. Ambiguous
strings fail. This repairs scorer formatting, never the semantic judgement.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def label(value):
    matches=re.findall(r'\b(covered|partial|missing)\b',value)
    if len(matches)!=1 or not (re.search(r'(?:^|[：:\s])(covered|partial|missing)$',value)
                              or re.match(r'^(covered|partial|missing)[：:]',value)):
        raise ValueError('Ambiguous point-status representation')
    return matches[0]


def normalize(original):
    result=deepcopy(original)
    for review in result.get('reviews',[]):
        if 'output' not in review:
            continue
        for assessment in review['output'].values():
            assessment['point_status']=[label(value) for value in assessment['point_status']]
    # Replacing canonical strings with their raw representation must recreate all
    # original bytes at the JSON-value level, including quotations and booleans.
    restored=deepcopy(result)
    for old,new in zip(original.get('reviews',[]), restored.get('reviews',[])):
        for arm, assessment in old.get('output',{}).items():
            new['output'][arm]['point_status']=assessment['point_status']
    assert restored==original
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((args.input/'summary.json').read_text(encoding='utf-8'))
    if summary['source_drift']:
        raise ValueError('Cannot repair a source-drift experiment')
    rows,hashes=[],{}
    gains,regressions,invalid=0,0,0
    for file in sorted(args.input.glob('*.json')):
        original=json.loads(file.read_text(encoding='utf-8'))
        if 'case' not in original:
            continue
        row=normalize(original)
        hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        (args.out/file.name).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        reviews=row['reviews']
        outcomes=[]
        valid=row['status']=='completed' and len(reviews)==2
        for review in reviews:
            if 'output' not in review:
                valid=False
                continue
            values={}
            for arm,a in review['output'].items():
                p=row['plans'][row['anonymous_mapping'][arm]]
                valid &= len(a['point_status'])==len(row['case']['required_points'])
                valid &= all(any(q in s['text'] for s in p['sources']) for q in a['supporting_quotes'])
                valid &= not a['sufficient'] or all(v=='covered' for v in a['point_status'])
                valid &= p['content_characters']<=1200
                values[row['anonymous_mapping'][arm]]=a['sufficient']
            regressions+=values['legacy'] and not values['hit_first_total_v1']
            outcomes.append(values['hit_first_total_v1'] and not values['legacy'])
        invalid+=not valid
        gains+=valid and len(outcomes)==2 and all(outcomes)
        rows.append({'id':row['case']['id'],'valid':valid,'outcomes':outcomes})
    if len(rows)!=6:
        raise ValueError('Six probes required')
    result={'kind':'format_only_rescoring_original_judgements_preserved','source_hashes':hashes,
            'semantic_changes':False,'paid_requests':0,'rows':rows,
            'both_judges_confirmed_gains':gains,'individual_regression_judgements':regressions,
            'invalid_cases':invalid,'candidate_accepted':gains>0 and regressions==0 and invalid==0,
            'criterion_unchanged':True,'human_semantic_validation':False}
    (args.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
