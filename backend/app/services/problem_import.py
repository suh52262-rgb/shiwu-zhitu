"""Local question authoring: private sources -> editable revision -> explicit publication.

Page previews are never student figures: they may contain answers or other questions.
Only operator-selected crops are copied into the published problem's figures directory.
Single-process authoring locks, consistent with the existing local demo architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pymupdf
from PIL import Image

from app.config import PROJECT_ROOT
from app.models.problem import Problem
from app.rag.exam_split import split_pages
from app.services.graph_versions import atomic_json, graph_lock
from app.services.problem_repository import get_problem_repository

MAX_BYTES = 30 * 1024 * 1024
MAX_PAGES = 30


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value or ''):
        raise ValueError('编号只允许字母、数字、下划线和短横线，长度1–80')
    if value.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
        raise ValueError('编号不能使用系统保留名称')
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def import_root():
    return Path(os.getenv('PROBLEM_IMPORTS_DIR') or PROJECT_ROOT / 'data' / 'problem_imports')


class ImportConflict(ValueError):
    pass


class ProblemImporter:
    def __init__(self, root=None, problems_root=None):
        self.root = Path(root) if root is not None else import_root()
        self.problems_root = Path(problems_root) if problems_root is not None else get_problem_repository().root

    def folder(self, import_id):
        return self.root / safe_id(import_id)

    def load(self, import_id):
        return json.loads((self.folder(import_id) / 'manifest.json').read_text(encoding='utf-8'))

    def list(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.root.glob('*/manifest.json'))]

    def create(self, data: bytes, filename: str, *, use_vision=False, reader=None):
        if not data or len(data) > MAX_BYTES:
            raise ValueError('文件为空或超过30MB')
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.pdf','.png','.jpg','.jpeg','.webp'}:
            raise ValueError('支持PDF、PNG、JPEG、WebP')
        iid = uuid4().hex
        folder = self.folder(iid)
        folder.mkdir(parents=True)
        (folder / ('source' + suffix)).write_bytes(data)
        record = {'schema_version':1, 'import_id':iid, 'filename':Path(filename).name,
                  'source_sha256':hashlib.sha256(data).hexdigest(), 'created_at':now(),
                  'status':'processing', 'pages':[], 'items':[], 'warnings':[], 'vision_requested':use_vision}
        atomic_json(folder/'manifest.json', record)
        try:
            if suffix == '.pdf':
                with pymupdf.open(stream=data, filetype='pdf') as doc:
                    if doc.needs_pass or not 0 < len(doc) <= MAX_PAGES:
                        raise ValueError('PDF需无密码且包含1–30页；较大文件请先拆分')
                    for index, page in enumerate(doc):
                        path = folder / f'page_{index+1}.png'
                        page.get_pixmap(dpi=120).save(path)
                        text = page.get_text('text', sort=True).strip()
                        record['pages'].append(self._page(index+1, text, path, use_vision, reader))
            else:
                with Image.open(folder/('source'+suffix)) as image:
                    image.load()
                    if image.width * image.height > 25_000_000:
                        raise ValueError('图片超过2500万像素')
                    image.convert('RGB').save(folder/'page_1.png')
                record['pages'].append(self._page(1, '', folder/'page_1.png', use_vision, reader))
            split_input = [(p['page'],p['text'].replace('【参考答案】','解：')) for p in record['pages']]
            drafts = split_pages(split_input)
            if not drafts:
                # A raw single page may be manually edited; multi-page papers are
                # not silently presented as one complete question.
                drafts = []
                record['warnings'].append('未自动切出题目；请按原页手工新增或分开题目，不代表原件没有题目')
            for draft in drafts:
                record['items'].append(self._item(draft.statement, draft.solution, draft.pages, draft.number))
            if not drafts and len(record['pages'])==1 and record['pages'][0]['text'].strip():
                raw=record['pages'][0]['text'].strip()
                if raw.startswith('【题目】'):
                    raw=raw[len('【题目】'):].strip()
                statement,marker,answer=raw.partition('【参考答案】')
                record['items'].append(self._item(statement.strip(),answer.strip() if marker else '',[1]))
                record['warnings'].append('单页未识别题号，整页作为候选草稿；若含多题请拆开后分别发布')
            record['status'] = 'draft'
            record['warnings'].append('自动切题与OCR均未经核对；公式、图文归属、跨页与双栏必须核对原页')
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = f'{type(exc).__name__}: {exc}'
        atomic_json(folder/'manifest.json', record)
        return record

    def _page(self, number, text, path, use_vision, reader):
        page = {'page':number, 'text':text, 'original_text':text, 'preview':path.name,
                'method':'text_layer' if text else 'unrecognized', 'status':'ready' if text else 'needs_transcription'}
        with Image.open(path) as image:
            page['preview_width'],page['preview_height']=image.size
        if use_vision:
            try:
                if reader is None:
                    from app.services.vision import call_vision
                    prompt = (PROJECT_ROOT/'prompts/problem_import/v1.md').read_text(encoding='utf-8')
                    output, meta = call_vision(path.read_bytes(), prompt)
                else:
                    output, meta = reader(path.read_bytes())
                if not output.strip():
                    raise ValueError('视觉模型返回空转写')
                page.update(text=output, original_text=output, method='vision', status='ready', model=meta,
                            prompt_version='problem_import/v1')
            except Exception as exc:
                page['vision_error'] = f'{type(exc).__name__}: {exc}'
                page['status'] = 'text_fallback' if text else 'failed'
        return page

    @staticmethod
    def _item(statement='', answer='', pages=None, number=None):
        return {'item_id':uuid4().hex, 'problem_id':'', 'title':'', 'statement':statement,
                'answer':answer, 'source_pages':pages or [], 'original_statement':statement,
                'original_answer':answer, 'exam_number':number, 'crops':[], 'revision':0,
                'history':[], 'review':None, 'publication':None}

    def add(self, iid):
        with graph_lock(self.folder(iid)):
            record=self.load(iid)
            item=self._item()
            record['items'].append(item)
            atomic_json(self.folder(iid)/'manifest.json',record)
            return item

    def revise(self, iid, item_id, base_revision):
        """Fork a published version. Old questions and Graphs stay immutable."""
        with graph_lock(self.folder(iid)):
            record = self.load(iid)
            parent = next(i for i in record['items'] if i['item_id'] == item_id)
            if parent['revision'] != base_revision or not parent['publication']:
                raise ImportConflict('需要当前已发布版本才能建立修订')
            # Response loss is recoverable without creating a second draft.
            existing = next((i for i in record['items']
                             if i.get('parent_item_id') == item_id), None)
            if existing:
                return existing
            pid = parent['publication']['problem_id']
            source = json.loads((self.problems_root/pid/'problem.json').read_text(encoding='utf-8'))
            if not self._matches_published(parent, source, self.content_hash(parent)):
                raise ImportConflict('旧发布内容已变化，不能据此创建修订')
            family = source.get('metadata', {}).get('question_family_id', pid)
            number = source.get('metadata', {}).get('question_revision', 1) + 1
            proposed = f'{family}_r{number}'
            if len(proposed) > 80 or (self.problems_root/proposed).exists():
                proposed = f'{family[:60]}_r{number}_{uuid4().hex[:8]}'
            new = self._item(parent['statement'], parent['answer'], list(parent['source_pages']))
            new.update(problem_id=safe_id(proposed), title=parent['title'],
                       crops=json.loads(json.dumps(parent['crops'])),
                       parent_item_id=item_id, parent_problem_id=pid,
                       parent_content_sha256=parent['publication']['content_sha256'],
                       question_family_id=family, question_revision=number)
            record['items'].append(new)
            atomic_json(self.folder(iid)/'manifest.json', record)
            return new

    def edit(self, iid, item_id, *, base_revision, values, reviewer='', confirmed=False):
        with graph_lock(self.folder(iid)):
            record=self.load(iid)
            item=next(i for i in record['items'] if i['item_id']==item_id)
            if item['publication']:
                raise ImportConflict('已发布草稿不可更改；请新建修订题目')
            if base_revision != item['revision']:
                raise ImportConflict('草稿已更新，请重新加载再核对')
            allowed={'problem_id','title','statement','answer','source_pages','crops'}
            if set(values)-allowed:
                raise ValueError('不支持的编辑字段')
            if item.get('parent_problem_id') and values.get('problem_id', item['problem_id']) != item['problem_id']:
                raise ImportConflict('修订题号已绑定版本，不能改写')
            candidate=dict(item,**values)
            available={p['page'] for p in record['pages']}
            if not candidate['source_pages'] or not set(candidate['source_pages']) <= available:
                raise ValueError('必须选择实际来源页')
            if not isinstance(candidate['crops'],list) or not all(isinstance(candidate[k],str) for k in ('problem_id','title','statement','answer')):
                raise ValueError('配图需为列表，题干、答案、标题和编号需为文字')
            self._check_crops(candidate['crops'],available,iid)
            if confirmed:
                safe_id(candidate['problem_id'])
                if not candidate['statement'].strip() or not reviewer.strip():
                    raise ValueError('发布复核需完整题干与复核者标识')
                if '[待核对]' in candidate['statement']+candidate['answer']:
                    raise ValueError('请先处理待核对标记')
            item['history'].append({k:item[k] for k in allowed} | {'revision':item['revision'],'saved_at':now(), 'review':item['review']})
            item.update(values)
            item['revision']+=1
            item['review']={'reviewer':reviewer.strip(),'date':now(),'content_sha256':self.content_hash(item)} if confirmed else None
            atomic_json(self.folder(iid)/'manifest.json',record)
            return item

    @staticmethod
    def content_hash(item):
        return digest({k:item[k] for k in ('problem_id','title','statement','answer','source_pages','crops')})

    def _check_crops(self,crops,pages,iid):
        for crop in crops:
            if set(crop)!={'page','bbox'} or crop['page'] not in pages or len(crop['bbox'])!=4:
                raise ValueError('配图必须指定来源页和四个像素坐标')
            x0,y0,x1,y1=crop['bbox']
            with Image.open(self.folder(iid)/f"page_{crop['page']}.png") as image:
                if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height:
                    raise ValueError('配图坐标越界或为空')

    def publish(self,iid,item_id,base_revision):
        with graph_lock(self.folder(iid)):
            record=self.load(iid)
            item=next(i for i in record['items'] if i['item_id']==item_id)
            if item['revision']!=base_revision:
                raise ImportConflict('发布版本过期')
            content_hash=self.content_hash(item)
            if not item['review'] or item['review']['content_sha256']!=content_hash:
                raise ImportConflict('需核对当前题干、答案和配图后再发布')
            pid=safe_id(item['problem_id'])
            target=self.problems_root/pid
            if item['publication']:
                if target.is_dir() and (target/'problem.json').is_file():
                    existing=json.loads((target/'problem.json').read_text(encoding='utf-8'))
                    if self._matches_published(item,existing,content_hash):
                        return item['publication']
                raise ImportConflict('发布记录与题库内容不一致')
            self.problems_root.mkdir(parents=True,exist_ok=True)
            with graph_lock(target):
                if target.exists():
                    # Recover a publication whose response/manifest write was interrupted.
                    existing=json.loads((target/'problem.json').read_text(encoding='utf-8')) if (target/'problem.json').is_file() else {}
                    meta=existing.get('metadata',{})
                    if meta.get('import_id')==iid and meta.get('import_item_id')==item_id and self._matches_published(item,existing,content_hash):
                        publication={'problem_id':pid,'date':meta['published_at'],'content_sha256':content_hash}
                        item['publication']=publication
                        atomic_json(self.folder(iid)/'manifest.json',record)
                        return publication
                    raise ImportConflict('题号已存在，未覆盖；请改用新题号')
                self._check_crops(item['crops'],set(item['source_pages']),iid)
                staging=Path(tempfile.mkdtemp(prefix='.import-',dir=self.problems_root))
                try:
                    figures=[]
                    for n,crop in enumerate(item['crops'],1):
                        relative=f'figures/figure_{n}.png'
                        (staging/'figures').mkdir(exist_ok=True)
                        with Image.open(self.folder(iid)/f"page_{crop['page']}.png") as image:
                            image.crop(tuple(crop['bbox'])).save(staging/relative)
                        figures.append(relative)
                    publication={'problem_id':pid,'date':now(),'content_sha256':content_hash}
                    problem=Problem(problem_id=pid,title=item['title'],statement=item['statement'],answer=item['answer'],figures=figures,
                                    metadata={'source':'reviewed_import','import_id':iid,'import_item_id':item_id,
                                              'source_sha256':record['source_sha256'],'source_pages':item['source_pages'],
                                              'source_crops':item['crops'],
                                              'import_content_sha256':content_hash,'published_at':publication['date'],
                                              'import_review':item['review'],
                                              'question_family_id':item.get('question_family_id',pid),
                                              'question_revision':item.get('question_revision',1),
                                              'parent_problem_id':item.get('parent_problem_id',''),
                                              'parent_content_sha256':item.get('parent_content_sha256','')})
                    atomic_json(staging/'problem.json',problem.model_dump())
                    staging.rename(target)
                finally:
                    if staging.exists():shutil.rmtree(staging)
                item['publication']=publication
                atomic_json(self.folder(iid)/'manifest.json',record)
                return publication

    @staticmethod
    def _matches_published(item,existing,content_hash):
        return (existing.get('metadata',{}).get('import_content_sha256')==content_hash
                and all(existing.get(k)==item[k] for k in ('problem_id','title','statement','answer'))
                and existing.get('metadata',{}).get('source_pages')==item['source_pages']
                and existing.get('metadata',{}).get('source_crops')==item['crops'])
