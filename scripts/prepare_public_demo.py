"""Install only the bundled AI-authored examples; never copy private demo data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def install(destination, source=None):
    source = Path(source or ROOT / 'examples/public_demo')
    destination = Path(destination).resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf8'))
    ids = manifest['problem_ids']
    if len(ids) != len(set(ids)) or any(not p.startswith('public_') or not p.replace('_', '').isalnum() for p in ids):
        raise ValueError('Invalid public demo IDs')
    expected = {f'problems/{pid}/{name}' for pid in ids for name in ('problem.json', 'graph.json')}
    if set(manifest['files']) != expected:
        raise ValueError('Public demo manifest does not describe exactly two files per problem')
    materials = {}
    for name, record in manifest['files'].items():
        path = source / name
        if path.is_symlink():
            raise ValueError('Public demo material must not be a symlink')
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Public demo material hash mismatch: ' + name)
        _, pid, leaf = name.split('/')
        if json.loads(data)['problem_id'] != pid:
            raise ValueError('Public demo problem identity mismatch')
        materials[pid, leaf] = data
    # Validate every conflict before creating any problem directory.
    existing = []
    for pid in ids:
        target = destination / pid
        if target.is_symlink():
            raise ValueError('Refusing symlink destination')
        if target.exists():
            if not target.is_dir() or any(not (target / name).is_file() or (target / name).is_symlink()
                    or (target / name).read_bytes() != materials[pid, name] for name in ('problem.json', 'graph.json')):
                raise ValueError('Existing problem differs; nothing overwritten: ' + pid)
            existing.append(pid)
    destination.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with tempfile.TemporaryDirectory(prefix='.public-demo-', dir=destination) as staging:
            for pid in ids:
                if pid in existing:
                    continue
                folder = Path(staging) / pid
                folder.mkdir()
                for name in ('problem.json', 'graph.json'):
                    (folder / name).write_bytes(materials[pid, name])
            for pid in ids:
                if pid not in existing:
                    # rename refuses replacement of a populated directory; never merge.
                    os.rename(Path(staging) / pid, destination / pid)
                    created.append(pid)
    except Exception:
        for pid in created:
            shutil.rmtree(destination / pid)
        raise
    return {'dataset_id': manifest['dataset_id'], 'installed': created, 'unchanged': existing,
            'graph_status': 'AI-authored, pending human review; legacy-demo only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/problems')
    args = parser.parse_args()
    print(json.dumps(install(args.destination), ensure_ascii=False, indent=2))
