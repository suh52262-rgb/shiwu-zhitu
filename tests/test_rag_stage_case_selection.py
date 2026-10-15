import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_rag_missing_point_stages import selected_stage_cases

Q={'cases':[{'id':'a','query':'first'},{'id':'b','query':'second'}]}
T={'cases':{'b':[{'id':'target'}]}}


def test_explicit_subset_never_masquerades_as_full_query_set():
    assert selected_stage_cases(Q,T)==[Q['cases'][1]]


@pytest.mark.parametrize('q,t',[
    ({'cases':[{'id':'a','query':''}]},T),
    ({'cases':[Q['cases'][0],Q['cases'][0]]},T),
    (Q,{'cases':{'unknown':[{'id':'target'}]}}),
    (Q,{'cases':{'b':[]}}),
    (Q,{'cases':{'b':[{},{}]}}),
    (Q,{'cases':{'b':[{'id':'target'},{'id':'target'}]}}),
])
def test_bad_identity_and_missing_targets_fail_instead_of_skipping(q,t):
    with pytest.raises(ValueError):selected_stage_cases(q,t)
