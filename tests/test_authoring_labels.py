import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontend'))
from authoring_labels import item_label


def test_same_title_versions_are_selectable_distinct_strings():
    parent={'title':'同一题','problem_id':'q','publication':{'problem_id':'q'}}
    revision={**parent,'problem_id':'q_r2','question_revision':2,'publication':None}
    assert item_label(parent,0)!=item_label(revision,1)
    assert 'v2' in item_label(revision,1)
    assert '草稿' in item_label(revision,1)


def test_unnamed_identical_drafts_still_have_distinct_labels():
    assert item_label({},0)!=item_label({},1)
