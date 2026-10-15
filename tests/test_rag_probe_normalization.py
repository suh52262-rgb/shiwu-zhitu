import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('rag_probe_normalization',Path(__file__).resolve().parents[1]/'scripts/normalize_rag_probe_reviews.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('value,expected',[('covered','covered'),('1: partial','partial'),('要点1：missing','missing'),('covered：原始解释','covered')])
def test_status_representation_only(value,expected):
    assert module.label(value)==expected


@pytest.mark.parametrize('value',['covered or missing','1: good','covered because','uncovered',''])
def test_ambiguous_or_new_meaning_rejected(value):
    with pytest.raises(ValueError):module.label(value)


def test_does_not_touch_verdict_quote_explanation_or_original():
    original={'reviews':[{'valid':False,'output':{'X':{'sufficient':True,'point_status':['1: covered'],
              'supporting_quotes':['逐字原文'],'explanation':'原始理由'}}}]}
    normalized=module.normalize(original)
    assert original['reviews'][0]['output']['X']['point_status']==['1: covered']
    assert normalized['reviews'][0]['valid'] is False
    assert normalized['reviews'][0]['output']['X']['supporting_quotes']==['逐字原文']
    assert normalized['reviews'][0]['output']['X']['sufficient'] is True
