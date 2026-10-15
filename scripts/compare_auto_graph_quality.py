"""Recompute from per-case records; unavailable checks are not failures or zeros."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.models.candidate_graph import CandidateGraph
from app.services.graph_generator import source_evidence
from app.services.graph_quality import inspect_quality

def arm(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    dataset=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    rows=[]
    for case in dataset['cases']:
        record=json.loads((folder/(case['id']+'.json')).read_text(encoding='utf-8'))
        row={'id':case['id'],'status':record['status'],'error':record.get('error'),
             'review':record.get('review'),'generation_sec':record.get('generation_sec')}
        if 'candidate' in record and manifest['version']=='v3':
            graph=CandidateGraph.model_validate(record['candidate']).to_problem_graph()
            row['source_and_coverage_check']=inspect_quality(graph,source_evidence(case['statement'],case['reference']),case['quantities'])
        rows.append(row)
    return {'total':len(rows),'completed':sum(r['status']=='completed' for r in rows),'rows':rows,
            'vocabulary_checked':sum('vocabulary' in json.loads((folder/(c['id']+'.json')).read_text(encoding='utf-8')) for c in dataset['cases']),
            'manifest':manifest}

def main():
    before=arm(sys.argv[1]);after=arm(sys.argv[2])
    assert before['manifest']['dataset_sha256']==after['manifest']['dataset_sha256']
    assert before['manifest']['rubric_sha256']==after['manifest']['rubric_sha256']
    common=[r['id'] for r in before['rows'] if r['status']=='completed' and any(s['id']==r['id'] and s['status']=='completed' for s in after['rows'])]
    fields=['serious_physics_errors','missing_quantities','false_alternative_paths','unsupported_assumptions']
    result={'before':before,'after':after,'paired_completed_ids':common,
            'paired_issue_counts':{name:{field:sum(len(r['review'][field]) for r in item['rows'] if r['id'] in common) for field in fields}
                                  for name,item in [('before',before),('after',after)]},
            'limitations':['same-family AI reviews, one generation per case','development probes, not holdout',
                           'original full-arm summaries counted absent vocabulary/rule checks as failures; use checked denominators here',
                           'failed cases retained in workflow denominator; do not call paired subset full-suite accuracy',
                           'source ID and path coverage checks do not establish semantic correctness','no human time measurement']}
    Path(sys.argv[3]).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'paired':common,'issue_counts':result['paired_issue_counts'],'completed':[before['completed'],after['completed']]},ensure_ascii=False))

if __name__=='__main__':main()
