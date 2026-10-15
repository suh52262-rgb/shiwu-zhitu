import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontend'))
from photo_job_state import matches, busy


def test_late_photo_cannot_override_switch_restart_or_new_answer():
    state={'attempt_id':'a','photo_epoch':2}
    p={'student_id':'s','problem_id':'p','base_attempt_id':'a','epoch':2,'status':'running'}
    state['photo_pending']=p
    assert busy(state,'s','p')
    assert not matches(state,p,'s','other')
    assert not matches({**state,'attempt_id':'new'},p,'s','p')
    assert not matches({**state,'photo_epoch':3},p,'s','p')
    assert matches({'photo_epoch':0},{**p,'epoch':0,'restored':True},'s','p')

def test_definite_rejection_is_distinct_from_unknown_outcome():
    import httpx
    from photo_job_state import submission_rejected
    for code in [408,429,500]:
        response=httpx.Response(code,request=httpx.Request('POST','http://local/photo/jobs'))
        try:response.raise_for_status()
        except httpx.HTTPStatusError as error:assert submission_rejected(error)==(429 if code==429 else None)
    assert submission_rejected(httpx.ReadTimeout('unknown')) is None
