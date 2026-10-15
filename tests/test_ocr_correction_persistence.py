"""Verify superseded media evidence against real PostgreSQL rows."""
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db_utils import db_available
from app.db import persistence
from app.db.postgres import get_engine
from app.db.tables import Attempt, HintRecord, Problem, Student
from app.models.problem import Problem as ProblemSchema


@pytest.mark.real_db
@pytest.mark.skipif(not db_available(), reason="PostgreSQL unavailable")
def test_superseded_ocr_hint_retained_but_never_resolved_as_effective():
    suffix = uuid4().hex[:12]
    student, problem, attempt = "ocr_student_" + suffix, "ocr_problem_" + suffix, "ocr_attempt_" + suffix
    source = "media_" + suffix
    try:
        assert persistence.persist_attempt_start(student, problem, attempt,
            problem=ProblemSchema(problem_id=problem, title="Synthetic OCR persistence fixture", content="Test only"))
        for label, meta in [("earlier", {}), ("original", {"source_id": source, "source_revision": 0}),
                            ("corrected", {"source_id": source, "source_revision": 1})]:
            assert persistence.persist_hint(attempt, 1, label, "concept", node="S1",
                                           state_before={"visited_nodes": [], **meta})
        assert persistence.supersede_media_hints(attempt, source, 0) == 1
        after = dict(visited_after=["S1"], current_node_after="S2", correct_after=True,
                     consecutive_failures_after=0)
        assert persistence.resolve_pending_hints(attempt, exclude_source_id=source, **after) == 1
        with Session(get_engine()) as session:
            hints = {r.hint: r for r in session.scalars(select(HintRecord).where(HintRecord.attempt_id == attempt))}
            assert len(hints) == 3  # delivered evidence was retained
            assert hints["original"].effective is None
            assert hints["original"].resolved_at is not None
            assert "superseded OCR" in hints["original"].effective_evidence
            assert hints["corrected"].resolved_at is None  # cannot judge itself
            assert hints["earlier"].effective is True
        assert persistence.resolve_pending_hints(attempt, **after) == 1
        with Session(get_engine()) as session:
            rows = session.scalars(select(HintRecord).where(HintRecord.attempt_id == attempt)).all()
            assert sum(r.effective is not None for r in rows) == 2
            assert next(r for r in rows if r.hint == "original").effective is None
    finally:
        with Session(get_engine()) as session, session.begin():
            session.execute(delete(Attempt).where(Attempt.id == attempt))
            session.execute(delete(Student).where(Student.id == student))
            session.execute(delete(Problem).where(Problem.id == problem))
