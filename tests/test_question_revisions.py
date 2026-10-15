import json

import pytest

from app.services.problem_binding import question_snapshot, student_view
from app.services.problem_import import ImportConflict
from app.services.problem_repository import ProblemRepository
from test_problem_import_workflow import create_reviewed, importer


def test_revision_has_new_identity_and_preserves_old_question_and_graph(importer):
    record, item = create_reviewed(importer)
    original_publication = importer.publish(record['import_id'], item['item_id'], 1)
    old_dir = importer.problems_root / item['problem_id']
    (old_dir / 'graph.json').write_text('old graph', encoding='utf-8')
    before = (old_dir / 'problem.json').read_bytes()
    revision = importer.revise(record['import_id'], item['item_id'], 1)
    assert revision == importer.revise(record['import_id'], item['item_id'], 1)
    assert revision['problem_id'] == 'new_question_r2'
    assert revision['review'] is None
    assert revision['publication'] is None
    with pytest.raises(ImportConflict):
        importer.publish(record['import_id'], revision['item_id'], 0)
    updated = importer.edit(record['import_id'], revision['item_id'], base_revision=0,
        values={'statement': 'Revised question', 'answer': 'private revised solution'},
        reviewer='test operator', confirmed=True)
    result = importer.publish(record['import_id'], updated['item_id'], 1)
    new_dir = importer.problems_root / result['problem_id']
    assert not (new_dir / 'graph.json').exists()
    assert (old_dir / 'problem.json').read_bytes() == before
    assert (old_dir / 'graph.json').read_text() == 'old graph'
    new = json.loads((new_dir / 'problem.json').read_text())
    assert new['metadata']['question_family_id'] == item['problem_id']
    assert new['metadata']['question_revision'] == 2
    assert new['metadata']['parent_content_sha256'] == original_publication['content_sha256']
    assert importer.revise(record['import_id'], updated['item_id'], 1)['problem_id'] == 'new_question_r3'


def test_stale_or_unpublished_cannot_revise(importer):
    record, item = create_reviewed(importer)
    with pytest.raises(ImportConflict):
        importer.revise(record['import_id'], item['item_id'], 1)
    importer.publish(record['import_id'], item['item_id'], 1)
    with pytest.raises(ImportConflict):
        importer.revise(record['import_id'], item['item_id'], 0)


def test_question_identity_tracks_answer_and_figure_content(importer):
    record, item = create_reviewed(importer)
    importer.publish(record['import_id'], item['item_id'], 1)
    repo = ProblemRepository(importer.problems_root)
    problem = repo.get_problem(item['problem_id'])
    snapshot, original = question_snapshot(problem, repo)
    assert snapshot['answer'] == problem.answer
    assert question_snapshot(problem.model_copy(update={'answer': 'changed'}), repo)[1] != original
    assert question_snapshot(problem.model_copy(update={'knowledge_points': ['new']}), repo)[1] == original
    path = repo.figure_path(problem.problem_id, problem.figures[0])
    path.write_bytes(b'changed image bytes')
    assert question_snapshot(problem, repo)[1] != original


def test_reference_answer_not_exposed_in_nested_student_state():
    private = {'state': {'problem_snapshot': {'answer': 'secret', 'statement': 'question'},
               'media_sources': [{'before_record': {'state': {'problem_snapshot': {'answer': 'secret'}}}}]}}
    public = student_view(private)
    assert 'secret' not in json.dumps(public)
    assert private['state']['problem_snapshot']['answer'] == 'secret'
    assert public['state']['problem_snapshot']['statement'] == 'question'
