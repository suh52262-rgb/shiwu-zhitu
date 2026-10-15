"""Source ownership, quote forms and legitimate non-quotation expressions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app.services.review_builder import unattributed_quotes, structural_defects, build_server_timeline

POOL = [
    {'id': 'S1', 'source': 'statement', 'text': '恒定电压电源，保持电压或电量不变。'},
    {'id': 'E1.1', 'source': 'student_answer', 'round': 1, 'text': '电压不变时，能量变少。'},
]


@pytest.mark.parametrize('opening,closing', [('「','」'),('『','』'),('“','”'),('‘','’'),('"','"'),("'","'")])
def test_wrong_source_is_rejected_even_when_fragment_exists_elsewhere(opening, closing):
    quote = opening + '电压不变' + closing
    assert unattributed_quotes('取题干给出的条件' + quote, POOL, strict_sources=True) == [quote]
    assert unattributed_quotes('你写的' + quote, POOL, strict_sources=True) == []


def test_terms_hypotheticals_and_mathematical_primes_are_not_verbatim_claims():
    for text in ['下一步选取“正方向”', '若原稿写的是「×」，请重新核对',
                 "你写的 v'=2 m/s 与 v''=3 m/s", "你写的 α'=2"]:
        assert unattributed_quotes(text, POOL, strict_sources=True) == []


def test_legacy_missing_source_is_unknown_but_current_policy_rejects_claim():
    text = '题目写明「向右运动」'
    assert unattributed_quotes(text, POOL[1:]) == []
    assert unattributed_quotes(text, POOL[1:], strict_sources=True) == ['「向右运动」']


def test_server_verbatim_timeline_does_not_reattribute_student_inner_quotes():
    pool = [{'id':'E1.1', 'source':'student_answer', 'round':1,
             'text':'我写：题目说“电压不变”，所以能量减少。'}]
    report = {'timeline':build_server_timeline({'items':pool}),
              'render_policy':'physical_check_only_v1'}
    assert not any(d['kind']=='unattributed_quote' for d in structural_defects(report, evidence_pool=pool))
    report['timeline'] = '题目说“电压不变”'
    assert any(d['kind']=='unattributed_quote' for d in structural_defects(report, evidence_pool=pool))
