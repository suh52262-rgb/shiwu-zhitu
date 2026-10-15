"""Inventory exact locked distributions and retain their installed license texts.

No package installation or network request. Missing licenses are explicit gaps,
not interpreted as permission. This is evidence collection, not legal clearance.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re


def collect(lock, destination, include_pip=True, supplements=None):
    lock, destination = Path(lock), Path(destination)
    if destination.exists():
        raise ValueError('Choose a new license evidence directory')
    requested = {}
    for line in lock.read_text(encoding='utf8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)', line.strip())
        if not match:
            raise ValueError('Expected exact name==version pins')
        requested[match[1]] = match[2]
    if include_pip:
        requested['pip'] = metadata.version('pip')
    distributions = []
    for name, version in sorted(requested.items(), key=lambda p: p[0].lower()):
        dist = metadata.distribution(name)
        if dist.version != version:
            raise ValueError('Installed distribution differs from the lock: ' + name)
        distributions.append((name, dist))
    destination.mkdir(parents=True)
    rows = []
    for name, dist in distributions:
        normalized = re.sub(r'[-_.]+', '-', name).lower()
        licenses, unavailable = [], []
        for f in dist.files or []:
            leaf = Path(str(f)).name
            if not leaf.lower().startswith(('license', 'licence', 'copying', 'notice', 'copyright')):
                continue
            source = dist.locate_file(f)
            if not source.is_file():
                unavailable.append({'distribution_file': str(f), 'reason': 'missing_installed_file'})
                continue
            content = source.read_bytes()
            if b'\0' in content:
                unavailable.append({'distribution_file': str(f), 'reason': 'binary_not_copied'})
                continue
            path = Path('licenses') / normalized / f'{len(licenses)+1:02d}-{leaf}'
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            licenses.append({'distribution_file': str(f), 'file': path.as_posix(),
                             'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})
        rows.append({'name': dist.metadata.get('Name', name), 'version': dist.version,
                     'license_expression': dist.metadata.get('License-Expression'),
                     'license_metadata': dist.metadata.get('License'),
                     'license_classifiers': [c for c in dist.metadata.get_all('Classifier', []) if 'License' in c],
                     'project_urls': dist.metadata.get_all('Project-URL', []),
                     'homepage': dist.metadata.get('Home-page'), 'license_files': licenses,
                     'unavailable_license_files': unavailable,
                     'status': 'license_text_retained_not_legal_clearance' if licenses else 'upstream_license_text_required'})
    if supplements:
        source = Path(supplements)
        additions = json.loads(source.read_text(encoding='utf8'))['files']
        for item in additions:
            if Path(item['file']).is_absolute() or '..' in Path(item['file']).parts:
                raise ValueError('Invalid relative license supplement path')
            original = source.parent / item['file']
            if original.is_symlink() or not original.resolve().is_relative_to(source.parent.resolve()):
                raise ValueError('License supplement escapes its recorded directory')
            content = original.read_bytes()
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('License supplement hash mismatch')
            row = next(r for r in rows if r['name'].lower() == item['distribution'].lower())
            target = destination / item['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            row['license_files'].append({'file': item['file'], 'sha256': item['sha256'],
                                         'bytes': len(content), 'source_url': item['url'],
                                         'origin': 'verified retained upstream text'})
            row['status'] = 'license_text_retained_not_legal_clearance'
    result = {'schema_version': 1, 'profile': 'exact Windows/Python3.11 deployment lock plus pip tool',
              'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
              'distribution_count': len(rows), 'distributions': rows,
              'gaps': [r['name'] for r in rows if not r['license_files']],
              'scope': 'installed distribution records only; not all native component obligations or author rights',
              'project_license': 'AGPL-3.0-only', 'models_and_datasets_distributed': False}
    (destination / 'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--supplements', type=Path)
    args = parser.parse_args()
    result = collect(args.lock, args.output, supplements=args.supplements)
    print(json.dumps({'distributions': result['distribution_count'], 'license_text_gaps': result['gaps']}))
