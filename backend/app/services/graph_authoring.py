"""Durable single-process graph authoring jobs for reviewed imports.

No automatic paid retries after restart. New candidates never replace an already
published teaching snapshot. AI checks/reviews never impersonate human approval.
"""
import hashlib
import json
import os
import queue
import threading
import logging
from pathlib import Path
from uuid import UUID, uuid4

from app.agent.prompt_registry import get_prompt, resolve_version
from app.config import PROJECT_ROOT
from app.config import get_settings
from app.models.problem_graph import ProblemGraph
from app.services.graph_generator import generate_candidate_graph,source_evidence
from app.services.graph_quality import inspect_quality,repairable,improvement
from app.services.graph_validator import validate_graph
from app.services.graph_versions import GraphVersions, atomic_json, graph_hash, graph_lock
from app.services.problem_import import ImportConflict, digest, now, safe_id
from app.services.problem_repository import get_problem_repository
from app.services.knowledge_vocabulary import canonical,validate as validate_vocabulary,vocabulary

ACTIVE={'queued','running'}
EDITABLE={'needs_review','needs_changes'}
logger=logging.getLogger(__name__)


class GraphJobBusy(ImportConflict):
    pass


class GraphAuthoring:
    def __init__(self,root=None,repository=None,generator=None,validator=None,start_worker=True):
        self.root=Path(root or os.getenv('GRAPH_JOBS_DIR') or PROJECT_ROOT/'data/graph_jobs')
        self.repo=repository or get_problem_repository()
        self.generator=generator or generate_candidate_graph
        self.validator=validator or validate_graph
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.pending=queue.Queue()
        for file in self.root.glob('*.json'):
            record=json.loads(file.read_text(encoding='utf-8'))
            if record['status'] in ACTIVE:
                record.update(status='interrupted',phase='interrupted',error='服务重启，生成结果可能部分存在；未自动重跑付费请求')
                atomic_json(file,record)
        if start_worker:
            threading.Thread(target=self._worker,daemon=True,name='graph-authoring').start()

    def path(self,jid):
        return self.root/(safe_id(jid)+'.json')

    def load(self,jid,pid):
        record=json.loads(self.path(jid).read_text(encoding='utf-8'))
        if record['problem_id']!=pid:raise ValueError('任务不属于当前题目')
        return record

    def listing(self,pid):
        safe_id(pid)
        with self.lock:
            records=[json.loads(p.read_text(encoding='utf-8')) for p in self.root.glob('*.json')]
        return sorted([r for r in records if r['problem_id']==pid],key=lambda r:r['created_at'],reverse=True)

    def _input(self,pid,options):
        safe_id(pid)
        problem=self.repo.get_problem(pid)
        if problem is None:raise FileNotFoundError('题目不存在')
        if problem.metadata.get('source')!='reviewed_import':
            raise ValueError('此入口仅用于已核对并发布的导入题目')
        if problem.figures and not options['figure_context'].strip():
            raise ValueError('带图题需填写已核对的图示说明；生成器只接收文字，不读取图片路径')
        provenance={}
        if options.get('figure_context_draft_id'):
            if not options.get('figure_context_confirmed'):
                raise ValueError('AI图示草稿必须先核对，再作为建图输入')
            from app.services.figure_context_drafts import get_figure_context_drafts
            draft=get_figure_context_drafts(self.repo).validate(pid,options['figure_context_draft_id'])
            provenance={'origin':'ai_vision_draft','request_id':draft['request_id'],'source':draft['source'],
                        'operator_confirmation':'self_reported_local_operator','semantic_validated':False,
                        'edited_from_draft':options['figure_context'].strip()!=draft['description'].strip()}
        figures={}
        for name in problem.figures:
            path=self.repo.figure_path(pid,name)
            if path is None:raise ValueError('题目登记的配图不存在')
            figures[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        roles=['graph_generator']+(['graph_validator'] if options['check_physics'] else [])
        prompts={r:{'version':resolve_version(r),'sha256':hashlib.sha256(get_prompt(r).encode()).hexdigest()} for r in roles}
        implementations={name:hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() for name in
            ('backend/app/services/graph_generator.py','backend/app/services/graph_validator.py','backend/app/models/candidate_graph.py','backend/app/services/graph_quality.py','backend/app/services/llm/factory.py')}
        settings=get_settings()
        model_config={field:getattr(settings,field) for field in
            ('llm_provider','llm_model','graph_generator_model','graph_validator_model','graph_generator_timeout')}
        from app.services.problem_binding import question_snapshot
        _, question_version = question_snapshot(problem, self.repo)
        return {'question_version':question_version, 'figure_context_provenance':provenance,
                'title':problem.title,'statement':problem.statement,'answer':problem.answer,'figures':figures,
                'course':problem.metadata.get('course','大学物理'),'options':options,'prompts':prompts,
                'implementations':implementations,'model_config':model_config,
                'vocabulary_sha256':hashlib.sha256((PROJECT_ROOT/'data/knowledge_vocabulary.json').read_bytes()).hexdigest()}

    def submit(self,pid,request_id,*,figure_context='',quantities_asked=None,check_physics=False,repair_once=True,
               figure_context_draft_id='',figure_context_confirmed=False):
        request_id=str(UUID(request_id))
        options={'figure_context':figure_context.strip(),'quantities_asked':list(dict.fromkeys(q.strip() for q in (quantities_asked or []) if q.strip())),
                 'check_physics':check_physics,'repair_once':repair_once,
                 'figure_context_draft_id':figure_context_draft_id,'figure_context_confirmed':figure_context_confirmed}
        captured=self._input(pid,options)
        fingerprint=digest(captured)
        with self.lock:
            records=[json.loads(p.read_text(encoding='utf-8')) for p in self.root.glob('*.json')]
            previous=next((r for r in records if r['client_request_id']==request_id),None)
            if previous:
                if previous['problem_id']!=pid or previous['input_sha256']!=fingerprint:
                    raise ImportConflict('同一请求编号的题目或生成输入已变化')
                return previous
            if any(r['problem_id']==pid and r['status'] in ACTIVE for r in records):
                raise GraphJobBusy('本题已有生成任务，请先查询原任务')
            if sum(r['status'] in ACTIVE for r in records)>=3:
                raise GraphJobBusy('建图队列已满，请稍后再提交')
            versions=GraphVersions(self.repo.problem_dir(pid))
            record={'job_id':uuid4().hex,'problem_id':pid,'client_request_id':request_id,'created_at':now(),
                    'status':'queued','phase':'queued','input_sha256':fingerprint,'input':captured,
                    'base_candidate':versions.manifest().get('candidate',''),'graph_version':'','revision':0,
                    'installed_candidate':versions.manifest().get('candidate',''),
                    'candidate':None,'graph':None,'validation':None,'history':[],'reviews':[],
                    'review':None,'publication':None,'error':'','timings':{}}
            record['model_calls']={}
            record['automatic_repairs']=[]
            atomic_json(self.path(record['job_id']),record)
            self.pending.put(record['job_id'])
            return record

    def _current(self,record):
        try:return digest(self._input(record['problem_id'],record['input']['options']))==record['input_sha256']
        except (ValueError,ImportConflict,FileNotFoundError):return False

    def _save(self,record):
        atomic_json(self.path(record['job_id']),record)

    def get_record_for_failure(self,jid):
        """Read the persisted partial output without recreating or rerunning a job."""
        try:
            record=json.loads(self.path(jid).read_text(encoding='utf-8'))
            return record if isinstance(record,dict) and record.get('job_id')==jid else None
        except (OSError,ValueError):
            logger.exception('无法读取Graph失败任务记录：%s',jid)
            return None

    def _worker(self):
        while True:
            jid=self.pending.get()
            try:self.run(jid)
            except Exception as exc:
                logger.exception('Graph任务失败，未自动重跑模型：%s',jid)
                with self.lock:
                    record=self.get_record_for_failure(jid)
                    if record and record.get('status') in ACTIVE:
                        record.update(status='failed',phase='failed',error='任务执行异常，保留已有输出，不自动重跑',error_type=type(exc).__name__)
                        try:self._save(record)
                        except Exception:
                            logger.exception('无法持久化Graph失败状态：%s',jid)
            finally:self.pending.task_done()

    def run(self,jid):
        import time
        with self.lock:
            record=json.loads(self.path(jid).read_text(encoding='utf-8'))
            if record['status']!='queued':return
            record.update(status='running',phase='generation')
            self._save(record)
        try:
            if not self._current(record):
                raise ImportConflict('题目或生成配置已变化，未调用模型')
            data=record['input']; options=data['options']; start=time.monotonic()
            statement=(data['title']+'\n'+data['statement']).strip()
            if options['figure_context']:
                statement+='\n【操作者核对的图示说明】\n'+options['figure_context']
            extra={}
            generation_client=None
            if self.generator is generate_candidate_graph:
                from app.services.llm import get_llm
                generation_client=get_llm('graph_generator')
                generation_client.reset_last_call()
                extra['llm']=generation_client
                extra['knowledge_vocabulary']=vocabulary()
                extra['quantities_asked']=options['quantities_asked']
            candidate=self.generator(problem_id=record['problem_id'],problem_statement=statement,
                                     reference_solution=data['answer'],course=data['course'],**extra)
            if generation_client and generation_client.last_call:
                record['model_calls']['generation']=generation_client.last_call.as_dict()
            record['candidate']=candidate.model_dump(mode='json')
            graph=candidate.to_problem_graph()
            graph.metadata['figure_context_provenance']=data.get('figure_context_provenance',{})
            for node in graph.nodes:node.knowledge_points=list(dict.fromkeys(canonical(k) for k in node.knowledge_points))
            graph.metadata['knowledge_points']=list(dict.fromkeys(canonical(k) for k in graph.metadata.get('knowledge_points',[])))
            graph.metadata.update(needs_review=True,authoring_job=jid,input_sha256=record['input_sha256'],
                                  question_version=record['input']['question_version'],
                                  quantities_asked=options['quantities_asked'],figure_context=options['figure_context'])
            evidence=source_evidence(statement,data['answer'])
            graph.metadata['source_evidence']=evidence
            record['graph']=graph.model_dump(mode='json')
            record['timings']['generation_sec']=round(time.monotonic()-start,3)
            record['phase']='validation'
            self._save(record)
            problem=self.repo.get_problem(record['problem_id']).model_copy(deep=True)
            problem.statement=statement
            start=time.monotonic()
            extra={};validation_client=None
            if self.validator is validate_graph and options['check_physics']:
                from app.services.llm import get_llm
                validation_client=get_llm('graph_validator')
                validation_client.reset_last_call()
                extra['llm']=validation_client
            report=self.validator(graph,problem=problem,use_llm=options['check_physics'],quantities_asked=options['quantities_asked'],**extra)
            if validation_client and validation_client.last_call:
                record['model_calls']['physics_validation']=validation_client.last_call.as_dict()
            record['timings']['validation_sec']=round(time.monotonic()-start,3)
            record['vocabulary']=self._vocabulary(graph)
            quality=inspect_quality(graph,evidence,options['quantities_asked'],require_grounding=self.generator is generate_candidate_graph)
            record['quality']=quality
            if options.get('repair_once',True) and repairable(quality,report) and self.generator is generate_candidate_graph:
                record['phase']='targeted_repair'
                self._save(record)
                attempt={'before_graph':graph.model_dump(mode='json'),'before_validation':report.model_dump(mode='json'),
                         'before_quality':quality,'accepted':False,'error':''}
                try:
                    repair_start=time.monotonic()
                    generation_client.reset_last_call()
                    revised=self.generator(problem_id=record['problem_id'],problem_statement=statement,reference_solution=data['answer'],
                        course=data['course'],llm=generation_client,knowledge_vocabulary=vocabulary(),quantities_asked=options['quantities_asked'],
                        correction={'candidate':candidate.model_dump(mode='json'),'quality_findings':quality['issues'],
                                    'validation_findings':[i.model_dump() for i in report.issues]})
                    if generation_client.last_call:record['model_calls']['targeted_repair']=generation_client.last_call.as_dict()
                    record['timings']['targeted_repair_generation_sec']=round(time.monotonic()-repair_start,3)
                    revised_graph=revised.to_problem_graph()
                    revised_graph.metadata['figure_context_provenance']=data.get('figure_context_provenance',{})
                    for node in revised_graph.nodes:node.knowledge_points=list(dict.fromkeys(canonical(k) for k in node.knowledge_points))
                    revised_graph.metadata['knowledge_points']=list(dict.fromkeys(canonical(k) for k in revised_graph.metadata.get('knowledge_points',[])))
                    revised_graph.metadata.update(authoring_job=jid,input_sha256=record['input_sha256'],needs_review=True,
                                                 question_version=record['input']['question_version'],
                        quantities_asked=options['quantities_asked'],figure_context=options['figure_context'],source_evidence=evidence)
                    extra={}
                    if validation_client:
                        validation_client.reset_last_call();extra['llm']=validation_client
                    repair_validation_start=time.monotonic()
                    revised_report=self.validator(revised_graph,problem=problem,use_llm=options['check_physics'],
                        quantities_asked=options['quantities_asked'],**extra)
                    record['timings']['repair_validation_sec']=round(time.monotonic()-repair_validation_start,3)
                    if validation_client and validation_client.last_call:
                        record['model_calls']['repair_physics_validation']=validation_client.last_call.as_dict()
                    revised_quality=inspect_quality(revised_graph,evidence,options['quantities_asked'])
                    attempt.update(after_graph=revised_graph.model_dump(mode='json'),after_validation=revised_report.model_dump(mode='json'),
                                   after_quality=revised_quality)
                    if improvement(quality,report,revised_quality,revised_report):
                        candidate,graph,report,quality=revised,revised_graph,revised_report,revised_quality
                        record['candidate']=candidate.model_dump(mode='json')
                        record['vocabulary']=self._vocabulary(graph)
                        record['quality']=quality
                        attempt['accepted']=True
                except Exception as exc:
                    attempt['error']=f'{type(exc).__name__}: {exc}'
                record['automatic_repairs'].append(attempt)
            graph.metadata['validation']=report.as_metadata()
            record.update(graph=graph.model_dump(mode='json'),validation=report.model_dump(mode='json'))
            with self.lock,graph_lock(self.repo.problem_dir(record['problem_id'])):
                versions=GraphVersions(self.repo.problem_dir(record['problem_id']))
                if not self._current(record) or versions.manifest().get('candidate','')!=record['base_candidate']:
                    raise ImportConflict('生成期间题目或候选版本已变化，产物保留但未挂到题库')
                # Invalid outputs remain editable in the job; do not install them.
                if report.valid and quality['ok']:
                    self.repo.save_graph(record['problem_id'],graph)
                    record['installed_candidate']=graph_hash(graph)
                record.update(status='needs_review' if report.valid and quality['ok'] else 'needs_changes',phase='complete',graph_version=graph_hash(graph))
                self._save(record)
        except Exception as exc:
            with self.lock:
                record.update(status='stale' if isinstance(exc,ImportConflict) else 'failed',phase='complete',
                              error=f'{type(exc).__name__}: {exc}')
                self._save(record)

    def edit(self,pid,jid,expected_version,raw_graph):
        with self.lock,graph_lock(self.repo.problem_dir(pid)):
            record=self.load(jid,pid)
            self._check(record,expected_version)
            graph=ProblemGraph.model_validate(raw_graph)
            if graph.problem_id!=pid:raise ValueError('Graph题号与当前题目不一致')
            # Governance and provenance cannot be edited via the JSON editor.
            graph.metadata={**record['graph']['metadata'],'needs_review':True}
            graph.metadata.pop('reviewed_by',None)
            graph.metadata.pop('reviewed_at',None)
            graph.metadata['node_evidence']=raw_graph.get('metadata',{}).get('node_evidence',{})
            graph.metadata['quantity_coverage']=raw_graph.get('metadata',{}).get('quantity_coverage',{})
            graph.metadata['open_questions']=raw_graph.get('metadata',{}).get('open_questions',[])
            for node in graph.nodes:node.knowledge_points=list(dict.fromkeys(canonical(k) for k in node.knowledge_points))
            graph.metadata['knowledge_points']=list(dict.fromkeys(k for n in graph.nodes for k in n.knowledge_points))
            report=self.validator(graph,problem=self.repo.get_problem(pid),use_llm=False,
                                  quantities_asked=record['input']['options']['quantities_asked'])
            graph.metadata['validation']=report.as_metadata()
            quality=inspect_quality(graph,graph.metadata.get('source_evidence',[]),record['input']['options']['quantities_asked'],
                                    require_grounding=self.generator is generate_candidate_graph)
            record['history'].append({'graph':record['graph'],'validation':record['validation'],
                                      'version':record['graph_version'],'revision':record['revision'],'date':now()})
            record.update(graph=graph.model_dump(mode='json'),validation=report.model_dump(mode='json'),
                          graph_version=graph_hash(graph),revision=record['revision']+1,review=None,
                          status='needs_review' if report.valid and quality['ok'] else 'needs_changes',quality=quality)
            record['vocabulary']=self._vocabulary(graph)
            if report.valid and quality['ok']:
                self.repo.save_graph(pid,graph)
                record['installed_candidate']=graph_hash(graph)
            self._save(record)
            return record

    def _check(self,record,expected_version):
        if record['status'] not in EDITABLE or record['publication']:
            raise ImportConflict('任务不可编辑或已发布；请明确重新生成')
        if not record['graph_version'] or record['graph_version']!=expected_version:
            raise ImportConflict('候选版本过期，请刷新后操作')
        if not self._current(record):raise ImportConflict('题目输入已变化，需重新生成')
        current=GraphVersions(self.repo.problem_dir(record['problem_id'])).manifest().get('candidate','')
        wanted=record['installed_candidate']
        if current!=wanted:raise ImportConflict('题库候选已被更新，当前任务不能覆盖它')

    def review(self,pid,jid,expected_version,reviewer,note,kind,confirmed):
        with self.lock,graph_lock(self.repo.problem_dir(pid)):
            record=self.load(jid,pid)
            self._check(record,expected_version)
            if not reviewer.strip() or not note.strip() or not confirmed:
                raise ValueError('需明确身份、复核说明并确认核对条件、所有小问、方向及路径')
            if not record['validation']['valid']:raise ValueError('先修复规则或物理校验中的严重问题')
            if kind=='human' and not self._vocabulary(ProblemGraph.model_validate(record['graph']))['ok']:
                raise ValueError('知识点含未确认的表外名称；请编辑为受控词表名称，再复核')
            if kind=='human' and not record.get('quality',{'ok':True})['ok']:
                raise ValueError('先处理来源、目标量覆盖或待核对输入问题')
            review={'reviewer':reviewer.strip(),'kind':kind,'note':note.strip(),'version':expected_version,'date':now()}
            record['reviews'].append(review)
            # Preserve AI opinions without clearing needs_review or creating a
            # human approval record in the existing governance mechanism.
            if kind=='human':
                self.repo.mark_reviewed(pid,reviewed_by=reviewer,note=note)
                record['review']=review
            elif kind!='ai':raise ValueError('复核类型必须为human或ai')
            self._save(record)
            return record

    def publish(self,pid,jid,expected_version,publisher):
        with self.lock,graph_lock(self.repo.problem_dir(pid)):
            record=self.load(jid,pid)
            if record['publication']:
                if expected_version!=record['graph_version']:raise ImportConflict('发布版本不一致')
                return record
            self._check(record,expected_version)
            if not record['review'] or record['review']['kind']!='human':
                raise ImportConflict('当前精确版本还没有人工复核记录；AI意见不代替人工批准')
            if not publisher.strip():raise ValueError('请填写发布者')
            if not self._vocabulary(ProblemGraph.model_validate(record['graph']))['ok']:
                raise ValueError('知识点不符合受控词表，未发布')
            versions=GraphVersions(self.repo.problem_dir(pid))
            version=expected_version if versions.manifest().get('published')==expected_version else self.repo.publish_graph(pid,published_by=publisher)
            self.repo.merge_knowledge_points(pid,self._vocabulary(ProblemGraph.model_validate(record['graph']))['known'])
            record['publication']={'version':version,'publisher':publisher,'date':now()}
            self._save(record)
            return record

    @staticmethod
    def _vocabulary(graph):
        points=list(graph.metadata.get('knowledge_points',[]))
        for node in graph.nodes:points.extend(node.knowledge_points)
        return validate_vocabulary(points)


_instances={}
_guard=threading.Lock()

def get_graph_authoring():
    root=Path(os.getenv('GRAPH_JOBS_DIR') or PROJECT_ROOT/'data/graph_jobs')
    repo=get_problem_repository()
    key=(str(root.resolve()),str(repo.root.resolve()))
    with _guard:
        if key not in _instances:_instances[key]=GraphAuthoring(root,repo)
        return _instances[key]
