"""Local authoring endpoints, separate from student question bank."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from app.services.problem_import import ImportConflict, MAX_BYTES, ProblemImporter

router=APIRouter(prefix='/problem-imports',tags=['problem authoring'])

class Edit(BaseModel):
    base_revision:int
    values:dict
    reviewer:str=''
    confirmed:bool=False

class Publish(BaseModel):
    base_revision:int

def invoke(fn,*args,**kwargs):
    try:
        return fn(*args,**kwargs)
    except ImportConflict as exc:
        raise HTTPException(409,detail=str(exc)) from exc
    except (FileNotFoundError,StopIteration) as exc:
        raise HTTPException(404,detail='导入记录或题目不存在') from exc
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(422,detail=str(exc)) from exc

@router.get('')
def listing():
    return {'items':ProblemImporter().list()}

@router.post('')
async def create(file:UploadFile=File(...),use_vision:bool=Form(False)):
    data=await file.read(MAX_BYTES+1)
    return await run_in_threadpool(invoke,ProblemImporter().create,data,file.filename or '',use_vision=use_vision)

@router.get('/{iid}')
def detail(iid:str):
    return invoke(ProblemImporter().load,iid)

@router.get('/{iid}/pages/{page}')
def page(iid:str,page:int):
    importer=ProblemImporter()
    record=invoke(importer.load,iid)
    if page not in {p['page'] for p in record['pages']}:
        raise HTTPException(404,detail='来源页不存在')
    return FileResponse(importer.folder(iid)/f'page_{page}.png',media_type='image/png')

@router.get('/{iid}/items/{item_id}/crops/{index}')
def crop_preview(iid:str,item_id:str,index:int):
    importer=ProblemImporter()
    record=invoke(importer.load,iid)
    item=next((i for i in record['items'] if i['item_id']==item_id),None)
    if item is None or index < 0 or index >= len(item['crops']):
        raise HTTPException(404,detail='配图不存在')
    crop=item['crops'][index]
    output=BytesIO()
    with Image.open(importer.folder(iid)/f"page_{crop['page']}.png") as image:
        image.crop(tuple(crop['bbox'])).save(output,format='PNG')
    return Response(output.getvalue(),media_type='image/png')

@router.post('/{iid}/items')
def add(iid:str):
    return invoke(ProblemImporter().add,iid)

@router.put('/{iid}/items/{item_id}')
def edit(iid:str,item_id:str,body:Edit):
    return invoke(ProblemImporter().edit,iid,item_id,**body.model_dump())

@router.post('/{iid}/items/{item_id}/publish')
def publish(iid:str,item_id:str,body:Publish):
    return invoke(ProblemImporter().publish,iid,item_id,body.base_revision)


@router.post('/{iid}/items/{item_id}/revise')
def revise(iid:str,item_id:str,body:Publish):
    return invoke(ProblemImporter().revise,iid,item_id,body.base_revision)
