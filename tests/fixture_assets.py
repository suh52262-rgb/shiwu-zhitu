"""Explicit synthetic regression assets, independent of the operator's data directory."""
from pathlib import Path
import shutil
import struct
import zlib
import hashlib
import json

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures/independent'
PROBLEMS = FIXTURE_ROOT / 'problems'
EVALUATION = FIXTURE_ROOT / 'evaluation'
T4_DATASET = EVALUATION / 't4_dataset.json'


def verify_fixture_bundle(root=FIXTURE_ROOT):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    expected = manifest['file_hashes']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.relative_to(root).as_posix() != 'manifest.json'}
    if actual != set(expected):
        raise ValueError('Independent fixture file set mismatch')
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Independent fixture hash mismatch: ' + name)
    return len(expected)


def install_bank(destination):
    """Copy into a test-owned directory. The tiny raster is only an asset-route sentinel."""
    destination = Path(destination)
    shutil.copytree(PROBLEMS, destination, dirs_exist_ok=False)
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    # Original, generated 8x8 blue square; not a physics illustration or student input.
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 8, 8, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + bytes([20, 70, 160]) * 8) * 8)) + chunk(b'IEND', b'')
    figure = destination / 'phy_002/figures/phy_002_fig1.png'
    figure.parent.mkdir()
    figure.write_bytes(png)
    return destination
