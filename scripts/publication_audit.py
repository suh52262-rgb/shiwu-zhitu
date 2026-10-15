"""Local publication preflight. Findings never include secret values or excerpts.

Scans current tracked files, reachable Git blobs (--all, not reflogs/unreachable),
and an optional source ZIP. Binary/oversize content is reported, not called clean.
This is not a complete PII, licensing, or Git-host security review.
"""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

TOKEN_RULES = {
    'api_token_shape': re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}'),
    'github_token_shape': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})'),
    'private_key_header': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'credential_url_literal': re.compile(rb'(?:postgres(?:ql)?(?:\+psycopg)?|https?)://[^\s\x22\x27/<>:@]+:[^\s\x22\x27<>/@]+@'),
}
LIMIT = 20 * 1024 * 1024


def private_path(name):
    parts = Path(name.replace('\\', '/')).parts
    leaf = parts[-1].lower() if parts else ''
    if leaf == '.env' or (leaf.startswith('.env.') and not leaf.endswith('.example')):
        return 'environment_file'
    if any(p in {'video_runs', 'video_jobs', 'photo_jobs', 'llm_circuits', 'textbooks', 'artifacts', 'output', '.venv'} for p in parts):
        return 'runtime_or_private_material'
    if leaf.endswith(('.mp4', '.mov', '.pdf', '.pptx', '.zip', '.dump')):
        return 'asset_permission_review'
    return None


def scan_text(data, *, known=()):
    hits = []
    for rule, expression in TOKEN_RULES.items():
        for match in expression.finditer(data):
            hits.append({'rule': rule, 'line': data[:match.start()].count(b'\n')+1,
                         'value_fingerprint': hashlib.sha256(match.group()).hexdigest()[:16]})
    for secret in known:
        start = data.find(secret)
        if start >= 0:
            hits.append({'rule': 'known_local_credential', 'line': data[:start].count(b'\n')+1,
                         'value_fingerprint': hashlib.sha256(secret).hexdigest()[:16]})
    return hits


def scan_bytes(data, *, known=()):
    if len(data) > LIMIT:
        return {'content_status': 'oversize_not_scanned', 'findings': []}
    if b'\0' in data[:8192]:
        return {'content_status': 'binary_not_scanned', 'findings': []}
    return {'content_status': 'text_scanned', 'findings': scan_text(data, known=known)}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def audit(root, source_zip=None, include_history=True):
    root = Path(root).resolve()
    from dotenv import dotenv_values
    known = [v.encode() for k,v in dotenv_values(root/'.env').items()
             if v and len(v) >= 16 and any(x in k.upper() for x in ['API_KEY', 'ACCESS_TOKEN', 'SECRET'])]
    report = {'source_commit': git(root,'rev-parse','HEAD').decode().strip(),
              'scope': 'tracked workspace; reachable Git blobs; optional source candidate',
              'limitations': ['patterns and known current credentials only; unknown credentials and PII can be missed',
                              'binary/oversize bodies not scanned', 'reflogs/unreachable objects and external hosting settings not scanned',
                              'path names in rev-list are hints, not every historic path for each blob',
                              'no license decision or repository-history rewrite performed'],
              'workspace': [], 'history': [], 'source_zip': [], 'remote_credentials': [], 'counts': {}}
    for name in git(root,'ls-files','-z').decode('utf-8').split('\0'):
        if not name:
            continue
        file = root/name
        if not file.is_file():
            report['workspace'].append({'path':name,'content_status':'tracked_file_missing','findings':[]})
            continue
        result = scan_bytes(file.read_bytes(), known=known)
        reason = private_path(name)
        if result['findings'] or reason or result['content_status'] != 'text_scanned':
            report['workspace'].append({'path':name,'path_review':reason,**result})
    if include_history:
        objects = git(root,'rev-list','--objects','--all').decode('utf-8').splitlines()
        request = ''.join(line.split(' ',1)[0]+'\n' for line in objects).encode()
        metadata = subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=request,cwd=root).decode().splitlines()
        names = {line.split(' ',1)[0]:line.split(' ',1)[1] if ' ' in line else '' for line in objects}
        proc = subprocess.Popen(['git','cat-file','--batch'],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        blobs = 0; scanned = 0
        try:
            for row in metadata:
                oid, kind, size = row.split(); size=int(size)
                if kind != 'blob':
                    continue
                blobs += 1
                name = names.get(oid,''); reason=private_path(name)
                if size > LIMIT:
                    report['history'].append({'object':oid,'path_hint':name,'bytes':size,'path_review':reason,
                                              'content_status':'oversize_not_scanned','findings':[]})
                    continue
                proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
                header=proc.stdout.readline().decode().split(); actual=int(header[2])
                data=proc.stdout.read(actual); assert len(data)==actual; assert proc.stdout.read(1)==b'\n'
                result=scan_bytes(data,known=known)
                scanned += result['content_status']=='text_scanned'
                if result['findings'] or reason or result['content_status']!='text_scanned':
                    report['history'].append({'object':oid,'path_hint':name,'bytes':size,'path_review':reason,**result})
        finally:
            proc.stdin.close();proc.stdout.close();proc.wait(timeout=10)
        report['counts'].update(history_blobs=blobs,history_text_blobs_scanned=scanned,
                                reachable_commits=int(git(root,'rev-list','--all','--count')))
    remote = subprocess.run(['git','config','--get-regexp',r'remote\..*\.url'],cwd=root,capture_output=True)
    for line in remote.stdout.splitlines():
        if b'://' in line and re.search(rb'://[^/\s]+@',line):
            report['remote_credentials'].append({'rule':'remote_url_userinfo','value_fingerprint':hashlib.sha256(line).hexdigest()[:16]})
    if source_zip:
        with zipfile.ZipFile(source_zip) as archive:
            for name in archive.namelist():
                if name.endswith('/'):
                    continue
                relative=name.split('/',1)[1] if '/' in name else name
                result=scan_bytes(archive.read(name),known=known)
                reason=private_path(relative)
                if result['findings'] or reason or result['content_status']!='text_scanned':
                    report['source_zip'].append({'path':relative,'path_review':reason,**result})
        report['source_zip_sha256']=hashlib.file_digest(Path(source_zip).open('rb'),'sha256').hexdigest()
    for scope in ['workspace','history','source_zip']:
        report['counts'][scope+'_secret_findings']=sum(len(x['findings']) for x in report[scope])
        report['counts'][scope+'_path_reviews']=sum(bool(x.get('path_review')) for x in report[scope])
    report['publication_status']='review_required'
    report['license_file_present']=any((root/n).is_file() for n in ['LICENSE','LICENSE.md','LICENSE.txt','COPYING'])
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--source-zip')
    parser.add_argument('--no-history',action='store_true')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    target=Path(args.out)
    if target.exists():
        raise SystemExit('Use a new report path; do not overwrite prior findings.')
    result=audit(args.root,args.source_zip,not args.no_history)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'publication_status':result['publication_status'],'counts':result['counts'],
                      'license_file_present':result['license_file_present']},ensure_ascii=False,indent=2))
