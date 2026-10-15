"""Local teacher read model. No models, DDL, mutations or client file paths."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re

from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.postgres import get_engine
from app.db.tables import Attempt
from app.services.learning_profile import profile_signals
from app.services.video_review_versions import fingerprint

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'data/video_runs'
PHOTOS = ROOT / 'data/teacher_media'


def database_identity(row):
    return fingerprint({'database': str(row[0]), 'address': str(row[1] or ''), 'port': row[2]})


def environment():
    with Session(get_engine()) as session:
        row = session.execute(text('SELECT current_database(), inet_server_addr()::text, inet_server_port()')).one()
    return {'database_identity': database_identity(row), 'identity_kind': 'database_connection_metadata_v1',
            'scope': 'local_connection_check_not_teacher_authorization'}


def read_decisions():
    # Unlike decisions(), do not create a missing table from a GET request.
    from app.services.error_evidence_reviews import ErrorEvidenceReview, view
    try:
        with Session(get_engine()) as session:
            records = session.scalars(select(ErrorEvidenceReview).order_by(ErrorEvidenceReview.id)).all()
            return [view(row) for row in records], 'ready'
    except Exception:
        return [], 'unavailable'


def load_context(attempt_id, student_id, problem_id):
    with Session(get_engine()) as session:
        attempt = session.get(Attempt, attempt_id)
        if attempt is None or attempt.student_id != student_id or attempt.problem_id != problem_id:
            raise HTTPException(404, '作答不存在或归属不匹配')
        return {'attempt_id': attempt.id, 'student_id': attempt.student_id, 'problem_id': attempt.problem_id,
                'state': attempt.state or {}}


def scoped_errors(*, start=None, end=None, student_ids=None, attempt_id=None, problem_id=None):
    query = select(Attempt)
    if start is not None:
        query = query.where(Attempt.start_time >= start)
    if end is not None:
        query = query.where(Attempt.start_time < end + timedelta(days=1))
    if student_ids is not None:
        query = query.where(Attempt.student_id.in_(student_ids))
    if attempt_id:
        query = query.where(Attempt.id == attempt_id)
    if problem_id:
        query = query.where(Attempt.problem_id == problem_id)
    with Session(get_engine()) as session:
        attempts = session.scalars(query.order_by(Attempt.id).limit(20001)).all()
        if len(attempts) > 20000:
            raise HTTPException(413, '统计范围过大，请缩小范围；未返回部分统计冒充全量')
        states = [(a.id, a.state or {}) for a in attempts]
        people = {a.id: a.student_id for a in attempts}
        problems = {a.id: a.problem_id for a in attempts}
    decisions, status = read_decisions()
    signals = profile_signals(states, decisions=decisions)
    evidence = signals['error_evidence']
    for item in evidence:
        item.update(student_id=people[item['attempt_id']], problem_id=problems[item['attempt_id']],
                    active=status == 'ready' and item['review_status'] not in {'dismissed', 'resolved'})
        if status != 'ready':
            item['review_status'] = 'unavailable'
    counts = None if status != 'ready' else {
        state: sum(e['review_status'] == state for e in evidence)
        for state in ['unreviewed', 'confirmed', 'dismissed', 'resolved']}
    return {'review_store_status': status, 'evidence': evidence, 'counts': counts,
            'attempt_count': len(states), 'semantic_validated': False,
            'policy': 'paper_evidence_v1', 'legacy_attempt_count': sum(
                s.get('profile_policy_version') != 'paper_evidence_v1' for _, s in states)}


def safe_json(path, root):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Evidence path escaped directory')
    return json.loads(path.read_text(encoding='utf-8'))


def owned_runs(context):
    if not RUNS.exists():
        return
    for directory in sorted(RUNS.iterdir()):
        if not re.fullmatch('[0-9a-f]{32}', directory.name) or not directory.is_dir():
            continue
        try:
            saved = safe_json(directory / 'review_input.json', RUNS)
            data = saved['input']
            if fingerprint(data) != saved['fingerprint']:
                continue
            if all(data.get(k) == context[k] for k in ['attempt_id', 'student_id', 'problem_id']):
                yield directory, saved
        except (OSError, ValueError, KeyError, TypeError):
            continue


def current_material(data, state):
    sources = {str(s.get('source_id')): s.get('revision', 0) for s in state.get('media_sources') or []}
    return (all(sources.get(k) == v for k, v in data.get('source_revisions', {}).items())
            and len(state.get('timeline') or []) == data.get('timeline_length', 0)
            and (not 'timeline_snapshot' in data or state.get('timeline', []) == data['timeline_snapshot'])
            and (not 'media_source_ids' in data or list(sources) == data['media_source_ids']))


def reports(context):
    from app.api.video import _structural_defects, _validated_review_payload
    output, issues = [], []
    for directory, original in owned_runs(context):
        candidates = list((directory / 'review_reports').glob('*.json'))
        if not candidates and (directory / 'review_report.json').exists():
            candidates = [directory / 'review_report.json']
        for path in sorted(candidates):
            try:
                report = safe_json(path, RUNS)
                revision = report.get('input_revision', 0)
                material = original if revision == 0 else safe_json(directory / 'review_inputs' / f'{int(revision)}.json', RUNS)
                data = material['input']
                if (fingerprint(data) != material['fingerprint']
                        or any(data.get(k) != context[k] for k in ['attempt_id', 'student_id', 'problem_id'])
                        or report.get('review_input_fingerprint') != material['fingerprint']):
                    raise ValueError('Report material identity mismatch')
                payload = _validated_review_payload(report['review'])
                if _structural_defects(payload, data.get('source_revisions')):
                    raise ValueError('Report structural/reference validation failed')
                output.append({'run_id': directory.name, 'report_id': path.stem,
                               'input_revision': revision, 'source_revisions': data.get('source_revisions', {}),
                               'input_fingerprint': material['fingerprint'],
                               'generation_fingerprint': report.get('generation_fingerprint'),
                               'material_status': 'current' if current_material(data, context['state']) else 'superseded',
                               'generation_status': 'stored_not_regenerated', 'review': payload,
                               'validation_scope': 'structure_and_references_not_physics'})
            except (OSError, ValueError, KeyError, TypeError, HTTPException):
                issues.append({'run_id': directory.name, 'report_id': path.stem, 'status': 'unavailable_or_invalid'})
    return {'reports': output, 'issues': issues, 'read_only': True, 'model_requests': 0}


def image_catalogue(context):
    output = []
    for directory, _ in owned_runs(context):
        try:
            result = safe_json(directory / 'results.json', RUNS)
        except (OSError, ValueError, KeyError):
            continue
        for frame in result.get('frames') or []:
            name = frame.get('frame')
            if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
                continue
            path = (directory / 'frames' / name).resolve()
            if not path.is_relative_to((directory / 'frames').resolve()) or not path.is_relative_to(RUNS.resolve()):
                continue
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                eid = hashlib.sha256((context['attempt_id'] + '/' + directory.name + '/' + name).encode()).hexdigest()
                output.append({'evidence_id': eid, 'media_type': 'video', 'run_id': directory.name,
                               'timestamp_sec': frame.get('timestamp_sec'), '_path': path})
    allowed_sources = {s.get('source_id') for s in context['state'].get('media_sources') or []}
    if PHOTOS.exists():
        for directory in PHOTOS.iterdir():
            if directory.name not in allowed_sources or not re.fullmatch('[0-9a-f]{32}', directory.name):
                continue
            try:
                meta = safe_json(directory / 'meta.json', PHOTOS)
                if not all(meta.get(k) == context[k] for k in ['attempt_id', 'student_id', 'problem_id']):
                    continue
                path = (directory / meta['file']).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_relative_to(PHOTOS.resolve()) or not path.is_file():
                    continue
                output.append({'evidence_id': directory.name, 'source_id': directory.name,
                               'media_type': 'image', '_path': path})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return output


def retain_photo(raw, *, source_id, student_id, problem_id, attempt_id):
    if not re.fullmatch('[0-9a-f]{32}', source_id or ''):
        return 'unavailable'
    from io import BytesIO
    from PIL import Image
    from app.services.graph_versions import atomic_json
    try:
        with Image.open(BytesIO(raw)) as image:
            fmt = image.format
            image.verify()
        extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(fmt)
        if not extension:
            return 'unsupported_format'
        directory = PHOTOS / source_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / ('original' + extension)).write_bytes(raw)
        atomic_json(directory / 'meta.json', {'source_id': source_id, 'student_id': student_id,
                    'problem_id': problem_id, 'attempt_id': attempt_id, 'file': 'original' + extension})
        return 'retained'
    except (OSError, ValueError, Image.DecompressionBombError):
        return 'failed'


def evidence(context):
    decisions, status = read_decisions()
    signals = profile_signals([(context['attempt_id'], context['state'])], decisions=decisions)
    if status != 'ready':
        for item in signals['error_evidence']:
            item['review_status'] = 'unavailable'
    images = [{k: v for k, v in item.items() if k != '_path'} for item in image_catalogue(context)]
    return {**signals, 'review_store_status': status, **reports(context), 'images': images,
            'missing_image_status': 'not_retained_or_unavailable' if not images else None}
