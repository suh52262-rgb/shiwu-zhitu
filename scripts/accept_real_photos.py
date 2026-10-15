"""Four authorized retained full frames, real image API and correction lifecycle.

No second video decode. OCR formatting edits are AI lifecycle probes, not gold
transcriptions. Every case including failures is saved; no paid auto resubmission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from accept_real_videos import disposable_services, write


def freeze_frames(archives, out):
    cases=[]
    for pid, archive in archives.items():
        source=next(archive.glob('perception/*/results.json'))
        data=json.loads(source.read_text(encoding='utf-8'))
        frames=[f for f in data['frames'] if not f.get('roi_tag') and f.get('has_content')]
        end=max(f['timestamp_sec'] for f in frames)
        for label,ratio in [('middle',0.5),('late',0.9)]:
            frame=min(frames,key=lambda f:abs(f['timestamp_sec']-ratio*end))
            original=source.parent/'frames'/frame['frame']
            path=out/f'{pid}_{label}.jpg'
            shutil.copyfile(original,path)
            cases.append({'id':path.stem,'problem_id':pid,'path':str(path.resolve()),
                          'source_frame':str(original.resolve()),'timestamp_sec':frame['timestamp_sec'],
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo02',type=Path,required=True)
    parser.add_argument('--demo03',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',help='Focused recovery; original batch is retained, never replaced')
    args=parser.parse_args()
    if args.out.exists():parser.error('Use new output directory')
    args.out.mkdir(parents=True)
    cases=freeze_frames({'demo_02':args.demo02,'demo_03':args.demo03},args.out)
    if args.cases:
        if not set(args.cases).issubset({c['id'] for c in cases}):parser.error('Unknown recovery case')
        cases=[c for c in cases if c['id'] in args.cases]
    write(args.out/'manifest.json',{'cases':cases,'origin':'authorized video frames; real handwriting; no gold OCR',
        'selection':'nearest retained full content frame to 50% and 90% of last content timestamp',
        'source':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for base in (ROOT/'backend/app',ROOT/'prompts') for p in base.rglob('*') if p.suffix in {'.py','.md'}}})
    with disposable_services():
        from app.config import get_settings
        settings=get_settings(); settings.llm_provider='deepseek'; settings.llm_model='deepseek-flash'
        for role in ('step_matcher','analyze_step','diagnosis','tutor','review'):
            setattr(settings,role+'_model','deepseek-flash')
        from app.db.postgres import get_engine
        from app.db.tables import Base
        Base.metadata.create_all(get_engine())
        from fastapi.testclient import TestClient
        from app.main import app
        records=[]
        with TestClient(app) as client:
            for case in cases:
                sid='photo_accept_'+case['id']; started=time.perf_counter()
                with Path(case['path']).open('rb') as file:
                    response=client.post('/video/analyze_image',data={'student_id':sid,'problem_id':case['problem_id'],
                        'request_hint':'true','include_review':'false'},files={'file':('frame.jpg',file,'image/jpeg')})
                record={'case':case,'http_status':response.status_code,'elapsed_s':time.perf_counter()-started,
                        'response':response.json(),'edit_provenance':'AI whitespace-only lifecycle probe, not gold OCR'}
                write(args.out/f"{case['id']}.json",record)
                if response.status_code == 200:
                    data=response.json(); aid=data['attempt_id']; last=data['rounds'][-1]
                    before=client.get('/attempt/'+aid).json()
                    text=data['recognized_text']
                    corrected=text.replace('=', ' = ',1) if '=' in text else text.replace('\n','\n\n',1)
                    if corrected!=text:
                        body={'student_id':sid,'problem_id':case['problem_id'],'attempt_id':aid,
                              'source_id':last['source_id'],'base_revision':last['source_revision'],
                              'corrected_text':corrected}
                        correction=client.post('/video/correct_ocr',json=body)
                        record['correction']={'http_status':correction.status_code,'response':correction.json()}
                        if correction.status_code==200:
                            repeat=client.post('/video/correct_ocr',json=body)
                            record['duplicate_correction']={'http_status':repeat.status_code,'response':repeat.json()}
                    after=client.get('/attempt/'+aid).json()
                    record['before']=before;record['after']=after
                    help_response=client.post('/attempt/analyze',json={'student_id':sid,'problem_id':case['problem_id'],
                         'attempt_id':aid,'answer':'','request_hint':True})
                    record['help']={'http_status':help_response.status_code,'response':help_response.json()}
                    record['traces']=client.get('/attempt/'+aid+'/traces').json()
                    record['no_hidden_review']=data['review_status']=='not_requested' and data['review'] is None
                write(args.out/f"{case['id']}.json",record)
                records.append(record)
                print('COMPLETE '+case['id']+' HTTP '+str(response.status_code),flush=True)
        write(args.out/'summary.json',{'cases':len(records),'image_successes':sum(r['http_status']==200 for r in records),
            'correction_statuses':{r['case']['id']:r.get('correction',{}).get('http_status') for r in records},
            'help_statuses':{r['case']['id']:r.get('help',{}).get('http_status') for r in records},
            'review_exclusion_check':{'checked':sum(r['http_status']==200 for r in records),
                'not_requested':sum(r.get('no_hidden_review',False) for r in records),
                'unavailable':sum(r['http_status']!=200 for r in records)},
            'scope':'real API lifecycle observations; not OCR or teaching accuracy proof'})
        get_engine().dispose()


if __name__=='__main__':main()
