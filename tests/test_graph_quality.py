from app.models.candidate_graph import CandidateGraph
from app.services.graph_generator import source_evidence
from app.services.graph_quality import inspect_quality,improvement,repairable
from app.services.graph_validator import validate_graph

def graph():
    return CandidateGraph(problem_id='p',nodes=[
        {'id':'S1','name':'受力','type':'modeling','description':'only F'},
        {'id':'S2','name':'计算','type':'calculation','description':'a=4'},
        {'id':'S3','name':'结果','type':'result','description':'向右'}],
        edges=[['S1','S2'],['S2','S3']],valid_paths=[['S1','S2','S3']],
        node_evidence={'S1':['Q001'],'S2':['Q001','R001'],'S3':['R001']},
        quantity_coverage={'大小':['S2'],'方向':['S3']}).to_problem_graph()

def test_grounding_is_exact_catalogue_but_never_semantic_certification():
    evidence=source_evidence('题干原文\n第二行','参考原文')
    assert evidence[0]['text']=='题干原文' and evidence[2]['text']=='参考原文'
    result=inspect_quality(graph(),evidence,['大小','方向'])
    assert result['ok'] and result['semantic_validated'] is False

def test_wrong_source_and_missing_target_are_not_accepted():
    candidate=graph()
    candidate.metadata['node_evidence']['S2']=['R999']
    candidate.metadata['quantity_coverage'].pop('方向')
    result=inspect_quality(candidate,source_evidence('题干','参考'),['大小','方向'])
    assert {i['kind'] for i in result['issues']}=={'source_reference','quantity_coverage'}

def test_independent_questions_cannot_be_optional_paths():
    candidate=graph()
    candidate.edges=[['S1','S2'],['S1','S3']]
    candidate.valid_paths=[['S1','S2'],['S1','S3']]
    result=inspect_quality(candidate,source_evidence('题干','参考'),['大小','方向'])
    assert len([i for i in result['issues'] if i.get('path_index') is not None])==2

def test_model_cannot_hide_target_on_generic_modeling_node():
    candidate=graph()
    candidate.metadata['quantity_coverage']['大小']=['S1']
    assert not inspect_quality(candidate,source_evidence('题干','参考'),['大小'])['ok']

def test_input_conflict_is_left_for_review_not_automatically_repaired():
    candidate=graph()
    candidate.metadata['open_questions']=['参考答案8/2=2与题干算术矛盾']
    quality=inspect_quality(candidate,source_evidence('题干','参考'),['大小','方向'])
    assert not repairable(quality,validate_graph(candidate,use_llm=False))

def test_repair_cannot_win_by_skipping_previous_physics_check():
    candidate=graph()
    before=validate_graph(candidate,use_llm=False)
    after=before.model_copy(deep=True)
    before.checks['physics']='issues';after.checks['physics']='skipped'
    assert not improvement({'issues':[{'kind':'source_reference'}]},before,{'issues':[]},after)

def test_repair_strictly_reduces_defects_without_new_families():
    report=validate_graph(graph(),use_llm=False)
    assert improvement({'issues':[{'kind':'source_reference'}]},report,{'issues':[]},report)
    assert not improvement({'issues':[{'kind':'source_reference'}]},report,{'issues':[{'kind':'quantity_coverage'}]},report)

def test_malformed_editor_metadata_returns_findings_instead_of_crashing():
    candidate=graph();candidate.metadata['node_evidence']=[];candidate.metadata['quantity_coverage']=None
    assert not inspect_quality(candidate,source_evidence('题干','参考'),['大小'])['ok']
