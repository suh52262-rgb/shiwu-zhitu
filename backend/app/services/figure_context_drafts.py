"""Idempotent AI visual descriptions; never a published or verified graph input."""
import hashlib
import json
from pathlib import Path
import threading
from uuid import UUID,uuid4
from datetime import datetime,timezone

from app.config import PROJECT_ROOT,get_settings
from app.services.problem_import import ImportConflict
from app.services.problem_repository import get_problem_repository
from app.services.video_review_versions import fingerprint
from app.services.graph_versions import atomic_json

EPOCH=uuid4().hex
_services={}


class FigureContextDrafts:
    def __init__(self,root,repository,vision=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.repo=repository;self.vision=vision;self.lock=threading.RLock()

    def path(self,rid):return self.root/(str(UUID(rid))+'.json')

    def source(self,pid,index):
        problem=self.repo.get_problem(pid)
        if problem is None:raise FileNotFoundError()
        if index<0 or index>=len(problem.figures):raise ValueError('请选择本题已登记的配图')
        relative=problem.figures[index];path=self.repo.figure_path(pid,relative)
        if path is None:raise ValueError('登记配图不可读')
        raw=path.read_bytes()
        if len(raw)>12*1024*1024:raise ValueError('视觉草稿配图超过12MB，请先裁切')
        source={'problem_id':pid,'figure_index':index,'figure':relative,
                'figure_sha256':hashlib.sha256(raw).hexdigest(),
                'statement_sha256':hashlib.sha256(problem.statement.encode()).hexdigest()}
        return problem,raw,source

    def view(self,record):
        value={k:v for k,v in record.items() if k not in {'raw_response','owner_epoch'}}
        try:
            _,_,current=self.source(record['problem_id'],record['source']['figure_index'])
            value['stale']=current!=record['source']
        except (ValueError,FileNotFoundError):value['stale']=True
        value.update(review_required=True,semantic_validated=False,origin='ai_vision_draft')
        return value

    def get(self,pid,rid):
        with self.lock:
            record=json.loads(self.path(rid).read_text(encoding='utf-8'))
            if record['problem_id']!=pid:raise FileNotFoundError()
            if record['status']=='processing' and record.get('owner_epoch')!=EPOCH:
                record.update(status='interrupted',error='服务重启中断，不自动重跑付费请求')
                atomic_json(self.path(rid),record)
            return self.view(record)

    def listing(self,pid):
        values=[]
        for path in self.root.glob('*.json'):
            record=json.loads(path.read_text(encoding='utf-8'))
            if record['problem_id']==pid:values.append(self.get(pid,record['request_id']))
        return sorted(values,key=lambda x:x['created_at'],reverse=True)

    def submit(self,pid,rid,index=0):
        from app.agent.prompt_registry import get_prompt,resolve_version
        from app.services.vision import call_vision
        rid=str(UUID(rid));problem,raw,source=self.source(pid,index)
        prompt=get_prompt('figure_context')+'\n题面（仅用于定位，不替代图面标签）：\n'+problem.statement
        model=get_settings().graph_generator_model or 'deepseek-flash'
        signature=fingerprint({'source':source,'prompt':prompt,'model':model})
        with self.lock:
            if self.path(rid).exists():
                previous=json.loads(self.path(rid).read_text(encoding='utf-8'))
                if previous['input_sha256']!=signature:raise ImportConflict('同一视觉请求编号的图或参数已变化')
                return self.get(pid,rid)
            record={'request_id':rid,'problem_id':pid,'source':source,'input_sha256':signature,
                    'owner_epoch':EPOCH,'created_at':datetime.now(timezone.utc).isoformat(),
                    'status':'processing','description':'','uncertainties':[],'error':'',
                    'prompt_version':resolve_version('figure_context'),'model':model}
            atomic_json(self.path(rid),record)
        # Release the registry lock during paid IO: GET/recovery must stay usable.
        try:
            text,meta=(self.vision or call_vision)(raw,prompt,model=model,timeout=120)
            record['raw_response']=text;record['usage']=meta
            cleaned=text.strip()
            if cleaned.startswith('```'):cleaned=cleaned.split('```')[1].removeprefix('json').strip()
            value=json.loads(cleaned)
            if not isinstance(value.get('description'),str) or not value['description'].strip():
                raise ValueError('视觉模型没有返回描述')
            if not isinstance(value.get('uncertainties',[]),list) or not all(isinstance(x,str) for x in value.get('uncertainties',[])):
                raise ValueError('不确定项格式不合法')
            if len(value['description'])>12000:raise ValueError('描述超过输入上限')
            record.update(status='ready',description=value['description'],uncertainties=value.get('uncertainties',[]))
        except Exception as exc:
            record.update(status='failed',error='视觉草稿未生成成功；不自动重跑',error_type=type(exc).__name__)
        with self.lock:
            atomic_json(self.path(rid),record)
            return self.view(record)

    def validate(self,pid,rid):
        record=self.get(pid,rid)
        if record['status']!='ready' or record['stale']:raise ImportConflict('图示草稿未就绪或来源已变化，请重新核对')
        return record


def get_figure_context_drafts(repository=None):
    repository=repository or get_problem_repository()
    root=PROJECT_ROOT/'data/figure_context_drafts'
    key=(str(root),str(repository.root))
    if key not in _services:_services[key]=FigureContextDrafts(root,repository)
    return _services[key]
