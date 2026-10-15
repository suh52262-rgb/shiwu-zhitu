import importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('finance_batch',ROOT/'scripts/run_t4_batch.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_unexecuted_cases_are_explicitly_blocked_and_never_evaluated(tmp_path):
    stopped=False;calls=[]
    def execute(item):
        nonlocal stopped
        calls.append(item['case_id']);stopped=True
        return {'status':'failed','error':{'type':'LLMResponseError','status_code':402}}
    items=[{'case_id':f'case{i}','repeat':1} for i in range(3)]
    rows=runner.execute_cases(items,execute,tmp_path,'frozen',stop_when=lambda:stopped)
    assert calls==['case0']
    assert [r['status'] for r in rows]==['failed','blocked','blocked']
    assert all(r['executed'] is False for r in rows[1:])
    assert len(list((tmp_path/'runs').glob('*.json')))==3
