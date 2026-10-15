from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_position_intro_material import validate_review


def data():
    points=[{'status':'covered','evidence_ids':['X1'],'explanation':'supported'}]*2
    return {'X':{'points':deepcopy(points)},'Y':{'points':[{'status':'partial','evidence_ids':['Y1'],'explanation':'needs check'}]*2},'ocr_issues':[]}

FACTS={'X':{'X1':{'text':'unchanged original'}},'Y':{'Y1':{'text':'other original'}}}


def test_server_fills_original_evidence_and_keeps_partial_insufficient():
    result=validate_review(data(),FACTS)
    assert result['X']['sufficient'] and not result['Y']['sufficient']
    assert result['X']['points'][0]['server_evidence'][0]['text']=='unchanged original'


@pytest.mark.parametrize('change',['cross_arm','empty_covered','duplicate','missing_point','ocr_unknown','ocr_empty'])
def test_invalid_reviews_and_ocr_sources_are_rejected(change):
    d=data()
    if change=='cross_arm':d['X']['points'][0]['evidence_ids']=['Y1']
    elif change=='empty_covered':d['X']['points'][0]['evidence_ids']=[]
    elif change=='duplicate':d['X']['points'][0]['evidence_ids']=['X1','X1']
    elif change=='missing_point':d['X']['points']=d['X']['points'][:1]
    elif change=='ocr_unknown':d['ocr_issues']=[{'arm':'X','evidence_ids':['Y1']}]
    else:d['ocr_issues']=[{'arm':'X','evidence_ids':[]}]
    with pytest.raises(ValueError):validate_review(d,FACTS)
