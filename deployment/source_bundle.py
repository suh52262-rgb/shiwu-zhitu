"""Prepare a local GitHub source candidate; never pushes or includes data assets."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from publication_audit import scan_bytes, private_path

# Label registries are project configuration, not教材正文、学生记录或索引数据。
PUBLIC_CONFIG = {'data/knowledge_vocabulary.json', 'data/document_names.json'}


def validate_entries(entries, known, rules):
    reviewed={(r['path'],r['rule'],r['value_fingerprint']) for r in rules.get('reviewed_examples',[])}
    accepted=[]
    for name,value in entries.items():
        if (Path(name).parts[0] in {'data','models','wheels','artifacts','output'} and name not in PUBLIC_CONFIG) or private_path(name):
            raise ValueError('Private/material path in source candidate: '+name)
        if name in PUBLIC_CONFIG:
            config = json.loads(value)
            if not isinstance(config, dict):
                raise ValueError('Invalid public configuration: ' + name)
            allowed = ({'version','note','how_to_add','concepts','aliases','dropped'}
                       if name.endswith('knowledge_vocabulary.json') else {'schema_version','note','documents'})
            if set(config) - allowed:
                raise ValueError('Unexpected content in public label configuration: ' + name)
            if name.endswith('document_names.json'):
                if config.get('schema_version') != 1 or not isinstance(config.get('documents'), dict):
                    raise ValueError('Invalid document-name registry')
                for row in config['documents'].values():
                    if not isinstance(row, dict) or set(row) - {'title','source_filename'} or any(not isinstance(v,str) or len(v)>512 for v in row.values()):
                        raise ValueError('Only display labels belong in the public registry')
            elif not isinstance(config.get('concepts'), dict) or any(not isinstance(k,str) or len(k)>120 for k in config['concepts']):
                raise ValueError('Invalid controlled vocabulary')
        result=scan_bytes(value,known=known)
        if result['content_status']!='text_scanned':
            raise ValueError('Unreviewed binary/oversize source content: '+name)
        for hit in result['findings']:
            identity=(name,hit['rule'],hit['value_fingerprint'])
            if hit['rule']=='known_local_credential' or identity not in reviewed:
                raise ValueError('Unreviewed credential finding: '+name+' / '+hit['rule'])
            accepted.append({'path':name,**hit})
    return accepted


def build(root, output, lock, *, source_ref='HEAD', directory=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new source archive path')
    if not source_ref or source_ref.startswith('-'):
        raise ValueError('Invalid source ref')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', source_ref+'^{commit}'], cwd=root).decode().strip()
    target = Path(directory).resolve() if directory else None
    if target and target.exists():
        raise ValueError('Choose a new clean source directory')
    rows = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', commit], cwd=root).split(b'\0')
    tree = {}
    for row in filter(None, rows):
        meta, name = row.split(b'\t', 1)
        mode, kind, oid = meta.split()
        tree[name.decode('utf8')] = (mode, kind, oid)
    names = list(tree)
    source_roots = {'backend', 'frontend', 'prompts', 'scripts', 'deployment', 'tests', 'simulation', '.github', '.streamlit', 'third_party'}
    allowed = [n for n in names if n and Path(n).parts[0] in source_roots
               and not n.startswith(('deployment/freezes/', 'deployment/validation/'))
               and not n.startswith('tests/fixtures/branch_merges/')
               and n not in {'scripts/_measure_stmt_quote.py', 'scripts/_run_miscon_eval.py'}]
    allowed += [n for n in sorted(PUBLIC_CONFIG) if n in tree]
    allowed += [n for n in names if n.startswith('examples/public_demo/')]
    allowed += [n for n in names if n and Path(n).parts[0] == 'evaluation' and n.endswith('.py')]
    allowed += [n for n in names if n.endswith('.cmd') and '/' not in n]
    allowed += ['requirements.txt', 'pytest.ini', '.gitignore', 'THIRD_PARTY_NOTICES.md']
    allowed += [n for n in ['LICENSE','LICENSE.md','LICENSE.txt','NOTICE','COPYING','.gitattributes'] if n in tree]
    # Known credentials stay private; this is a check, not a complete security audit.
    from dotenv import dotenv_values
    secrets = [v.encode() for k,v in dotenv_values(root/'.env').items()
               if v and len(v)>=16 and any(x in k.upper() for x in ['API_KEY','ACCESS_TOKEN','SECRET'])]
    required = set(allowed) | {'deployment/.env.deploy.example', 'deployment/compose.yml',
                              'deployment/GitHub源码快速开始.md', 'deployment/publication_review_rules.json'}
    blobs = {}
    proc = subprocess.Popen(['git','cat-file','--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for name in sorted(required):
            mode, kind, oid = tree[name]
            if mode not in {b'100644', b'100755'} or kind != b'blob':
                raise ValueError('Non-regular source entry: '+name)
            proc.stdin.write(oid+b'\n'); proc.stdin.flush()
            header = proc.stdout.readline().split()
            data = proc.stdout.read(int(header[2]))
            if proc.stdout.read(1) != b'\n':
                raise ValueError('Invalid Git blob response')
            blobs[name] = data
    finally:
        proc.stdin.close(); proc.stdout.close(); proc.wait(timeout=10)
    entries = {n: blobs[n] for n in sorted(set(allowed))}
    entries['.env.example'] = blobs['deployment/.env.deploy.example']
    entries['docker-compose.yml'] = blobs['deployment/compose.yml']
    entries['requirements-deploy.lock.txt'] = Path(lock).read_bytes()
    entries['README.md'] = blobs['deployment/GitHub源码快速开始.md']
    entries['.gitignore'] += b'\n# Source candidate: generated/local problem library stays private.\n/data/problems/\n'
    rules=json.loads(blobs['deployment/publication_review_rules.json'].decode('utf8'))
    accepted=validate_entries(entries,secrets,rules)
    identity = {'kind': 'source_candidate_not_publicly_released',
                'source_commit': commit,
                'source_bytes': 'committed Git blobs; supplied dependency lock recorded below',
                'public_demo': 'examples/public_demo only; AI-authored; human review pending',
                'project_license': 'AGPL-3.0-only' if 'LICENSE' in entries else 'not_declared',
                'public_configuration': [n for n in sorted(PUBLIC_CONFIG) if n in entries],
                'excluded': ['all private data assets including legacy demo problems', 'textbooks and derived content', 'models', 'wheels', 'student history', '.env', 'output', 'artifacts', 'historical docs and screenshots', 'internal freeze and validation records', 'Git history'],
                'limitations': ['not a full data deployment', 'public release requires source/history/license review', 'no remote push performed'],
                'credential_scan': {'unreviewed_findings':0,'reviewed_example_findings':accepted,
                                    'scope':'known current keys and configured token/URL/key patterns; not complete PII clearance'},
                'files': {n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in entries.items()}}
    entries['SOURCE_MANIFEST.json'] = (json.dumps(identity,ensure_ascii=False,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in entries.items():
            archive.writestr('ShiwuZhitu-source/'+name,value)
    if target:
        target.mkdir(parents=True)
        for name, value in entries.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    output.with_suffix('.zip.sha256').write_text(hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n')
    print(json.dumps({'files':len(entries),'bytes':output.stat().st_size,'archive':str(output)},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out',required=True)
    parser.add_argument('--lock',required=True)
    parser.add_argument('--ref', default='HEAD', dest='source_ref')
    parser.add_argument('--directory', help='Optional new directory, without original Git history')
    args = parser.parse_args()
    build(args.root,args.out,args.lock,source_ref=args.source_ref,directory=args.directory)
