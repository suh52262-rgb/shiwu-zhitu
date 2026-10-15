from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from app.services.figure_context_drafts import get_figure_context_drafts
from app.services.problem_import import ImportConflict

router=APIRouter(prefix='/problems/{pid}/figure-context-drafts',tags=['graph authoring'])


class DraftRequest(BaseModel):
    client_request_id:str
    figure_index:int=Field(default=0,ge=0)


def call(fn,*args):
    try:return fn(*args)
    except FileNotFoundError:raise HTTPException(404,'题目或视觉草稿不存在')
    except ImportConflict as exc:raise HTTPException(409,str(exc))
    except ValueError as exc:raise HTTPException(422,str(exc))


@router.post('')
def submit(pid:str,body:DraftRequest):return call(get_figure_context_drafts().submit,pid,body.client_request_id,body.figure_index)


@router.get('')
def listing(pid:str):return {'items':call(get_figure_context_drafts().listing,pid)}


@router.get('/{rid}')
def get(pid:str,rid:str):return call(get_figure_context_drafts().get,pid,rid)
