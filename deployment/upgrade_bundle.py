"""Create a new patch release from manifested files only; never copy local runtime.

Only explicitly named source files change. Data/model/wheel bytes stay unchanged.
Run finalize_bundle.py after validation to refresh fingerprints and build the ZIP.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from deployment_bundle import verify, write


def upgrade(previous, destination, version, files):
    previous, destination = Path(previous).resolve(), Path(destination).resolve()
    verify(previous)
    manifest = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid manifest path')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / relative, target)
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid patch path')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    identity = json.loads((destination / 'VERSION.json').read_text(encoding='utf-8'))
    identity.update(version=version, created_at=datetime.now().isoformat(),
                    previous_version=manifest['version'], patch_files=files,
                    source='previous manifested release plus explicit current workspace patch')
    identity.pop('source_tree_sha256', None)
    write(destination / 'VERSION.json', identity)
    print('PATCH_STAGED', destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--file', action='append', required=True)
    args = parser.parse_args()
    upgrade(args.previous, args.out, args.version, args.file)
