"""Reviewed A runtime snapshots for D's unchanged isolated experiment scope."""
from pathlib import Path
import hashlib

SUPPORTED_A_RUNTIMES={
    '1c6ecd31f61857196126eb8c9eb0d015856f2007704b95681fd1f20e210bf5e2':'a4acc6f00f1b540c26676942299e742ff886a591',
    'abb830c145fd422f44faad583569cdb8aa508af47aff88ab804b558c53db0f21':'3446dc08902be751a1b7e82403a0cd6f70cede37',
}


def require_supported_runtime(path=None):
    path=Path(path) if path is not None else Path(__file__).resolve().parents[1]/'runtime.py'
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual not in SUPPORTED_A_RUNTIMES:
        raise ValueError('unreviewed A runtime; use a validated A/D combined workspace')
    return {'runtime_sha256':actual,'reviewed_A_commit':SUPPORTED_A_RUNTIMES[actual]}
