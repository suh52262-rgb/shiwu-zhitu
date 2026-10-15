"""Append-only local operator decisions, separate from immutable writing/model state."""
from datetime import datetime,timezone
from uuid import UUID
from sqlalchemy import Integer,String,Text,JSON,DateTime,ForeignKey,select
from sqlalchemy.orm import Mapped,mapped_column,Session
from app.db.tables import Base,Attempt
from app.db.postgres import get_engine
from app.services.learning_profile import evidence_id
from app.services.video_review_versions import fingerprint


class ErrorEvidenceReview(Base):
    __tablename__='error_evidence_reviews'
    id:Mapped[int]=mapped_column(Integer,primary_key=True,autoincrement=True)
    request_id:Mapped[str]=mapped_column(String(36),unique=True)
    student_id:Mapped[str]=mapped_column(ForeignKey('students.id',ondelete='CASCADE'),index=True)
    attempt_id:Mapped[str]=mapped_column(ForeignKey('attempts.id',ondelete='CASCADE'),index=True)
    evidence_id:Mapped[str]=mapped_column(String(64),index=True)
    decision:Mapped[str]=mapped_column(String(16))
    actor_label:Mapped[str]=mapped_column(String(80))
    note:Mapped[str]=mapped_column(Text,default='')
    input_sha256:Mapped[str]=mapped_column(String(64))
    payload:Mapped[dict]=mapped_column(JSON)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=lambda:datetime.now(timezone.utc))


def ensure_store():ErrorEvidenceReview.__table__.create(get_engine(),checkfirst=True)


def view(row):
    timestamp=row.created_at
    if timestamp.tzinfo is None:timestamp=timestamp.replace(tzinfo=timezone.utc)
    return {'review_id':row.id,'request_id':row.request_id,'student_id':row.student_id,
            'attempt_id':row.attempt_id,'evidence_id':row.evidence_id,'decision':row.decision,
            'actor_label':row.actor_label,'note':row.note,'created_at':timestamp.astimezone(timezone.utc).isoformat(),
            'actor_authentication':'self_reported_local_operator','semantic_validated':False}


def decisions(student_id=None):
    ensure_store()
    with Session(get_engine()) as session:
        query=select(ErrorEvidenceReview).order_by(ErrorEvidenceReview.id)
        if student_id is not None:query=query.where(ErrorEvidenceReview.student_id==student_id)
        return [view(row) for row in session.scalars(query).all()]


def append(student_id,attempt_id,eid,decision,actor_label,note,request_id):
    if decision not in {'confirmed','dismissed','resolved'}:raise ValueError('复核动作不合法')
    if not actor_label.strip() or len(actor_label)>80 or len(note)>3000:raise ValueError('请提供有效的本地操作者标识和说明')
    request_id=str(UUID(request_id));ensure_store()
    payload={'student_id':student_id,'attempt_id':attempt_id,'evidence_id':eid,
             'decision':decision,'actor_label':actor_label.strip(),'note':note}
    identity=fingerprint(payload)
    with Session(get_engine()) as session,session.begin():
        # Lock the owning attempt so OCR replacement cannot race confirmation.
        attempt=session.scalar(select(Attempt).where(Attempt.id==attempt_id,Attempt.student_id==student_id).with_for_update())
        if attempt is None:raise FileNotFoundError('作答不存在或不属于当前学生')
        prior=session.scalar(select(ErrorEvidenceReview).where(ErrorEvidenceReview.request_id==request_id))
        if prior:
            if prior.input_sha256!=identity:raise ValueError('同一请求编号不能修改复核动作')
            return view(prior)
        state=attempt.state or {}
        if state.get('profile_policy_version')!='paper_evidence_v1':raise LookupError('历史标签没有当前书面证据，不能直接确认')
        snapshot=next((e for e in state.get('error_evidence') or []
                       if all(e.get(k) for k in ('concept','issue','student_text')) and evidence_id(attempt_id,e)==eid),None)
        if snapshot is None:raise LookupError('该错误证据已变化或被更正，请刷新后重新核对')
        row=ErrorEvidenceReview(**payload,request_id=request_id,input_sha256=identity,
            payload={'evidence_snapshot':snapshot,'origin':'local_operator_review_not_teacher_authentication'})
        session.add(row);session.flush();result=view(row)
    return result
