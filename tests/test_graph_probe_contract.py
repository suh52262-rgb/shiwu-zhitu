import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from probe_graph_teaching import review_request
from app.api.video import VideoReviewRequest


def test_probe_preflight_and_actual_review_use_same_production_schema():
    body=review_request('student','problem','attempt','0'*32)
    assert VideoReviewRequest.model_validate(body).run_id=='0'*32
    assert 'expected_revision' not in body
    with pytest.raises(ValidationError):review_request('student','problem','attempt','case-name-not-a-run-id')
