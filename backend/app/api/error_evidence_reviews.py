from typing import Literal
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from app.services import error_evidence_reviews as service
from app.db import persistence
from sqlalchemy.exc import SQLAlchemyError

router=APIRouter(prefix='/teacher/students/{sid}/error-evidence',tags=['teacher'])


class Decision(BaseModel):
    attempt_id:str
    decision:Literal['confirmed','dismissed','resolved']
    actor_label:str=Field(min_length=1,max_length=80)
    note:str=Field(default='',max_length=3000)
    client_request_id:str


@router.get('')
def listing(sid:str):
    try:
        profile=persistence.get_student_profile(sid)
        if profile is None:raise HTTPException(404,'学生不存在')
        return {'items':profile['error_evidence'],'review_store_status':profile.get('review_store_status','unknown'),
                'history':service.decisions(sid),'scope':'本地操作者复核，不验证教师身份或认知状态'}
    except (persistence.PersistenceUnavailable,SQLAlchemyError):raise HTTPException(503,'复核数据暂不可用')


@router.post('/{eid}/decisions')
def decide(sid:str,eid:str,body:Decision):
    try:return service.append(sid,body.attempt_id,eid,body.decision,body.actor_label,body.note,body.client_request_id)
    except FileNotFoundError as exc:raise HTTPException(404,str(exc))
    except LookupError as exc:raise HTTPException(409,str(exc))
    except ValueError as exc:raise HTTPException(409,str(exc))
    except SQLAlchemyError:raise HTTPException(503,'复核写入暂不可用；请先查询记录，不要直接重复确认')
