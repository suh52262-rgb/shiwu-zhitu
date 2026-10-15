from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.models.problem_graph import ProblemGraph
from app.services.graph_authoring import get_graph_authoring,GraphJobBusy
from app.services.problem_import import ImportConflict
from app.services.problem_repository import get_problem_repository
from app.services.knowledge_vocabulary import vocabulary

router=APIRouter(prefix='/problems/{pid}/graph-jobs',tags=['graph authoring'])

class Generate(BaseModel):
    client_request_id:str
    figure_context:str=Field(default='',max_length=12000)
    quantities_asked:list[str]=Field(default_factory=list,max_length=30)
    check_physics:bool=False
    repair_once:bool=True
    figure_context_draft_id:str=''
    figure_context_confirmed:bool=False

class Edit(BaseModel):
    expected_version:str
    graph:ProblemGraph

class Review(BaseModel):
    expected_version:str
    reviewer:str
    note:str
    kind:Literal['human','ai']='human'
    confirmed:bool=False

class Publish(BaseModel):
    expected_version:str
    publisher:str

def call(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except GraphJobBusy as exc:raise HTTPException(429,str(exc)) from exc
    except ImportConflict as exc:raise HTTPException(409,str(exc)) from exc
    except FileNotFoundError as exc:raise HTTPException(404,'题目或任务不存在') from exc
    except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc

@router.post('')
def generate(pid:str,body:Generate):
    return call(get_graph_authoring().submit,pid,body.client_request_id,
                figure_context=body.figure_context,quantities_asked=body.quantities_asked,check_physics=body.check_physics,repair_once=body.repair_once,
                figure_context_draft_id=body.figure_context_draft_id,figure_context_confirmed=body.figure_context_confirmed)

@router.get('')
def listing(pid:str):
    items=call(get_graph_authoring().listing,pid)
    release=get_problem_repository().graph_release(pid)
    return {'items':items,'teaching_version':release['graph_version'],'graph_status':release['graph_status'],
            'knowledge_vocabulary':vocabulary()}

@router.get('/{jid}')
def detail(pid:str,jid:str):
    return call(get_graph_authoring().load,jid,pid)

@router.put('/{jid}')
def edit(pid:str,jid:str,body:Edit):
    return call(get_graph_authoring().edit,pid,jid,body.expected_version,body.graph.model_dump(mode='json'))

@router.post('/{jid}/review')
def review(pid:str,jid:str,body:Review):
    return call(get_graph_authoring().review,pid,jid,body.expected_version,body.reviewer,body.note,body.kind,body.confirmed)

@router.post('/{jid}/publish')
def publish(pid:str,jid:str,body:Publish):
    return call(get_graph_authoring().publish,pid,jid,body.expected_version,body.publisher)
