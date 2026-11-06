"""Record the actual checkout and bytes used, without requiring git on PATH."""
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkout_commit(source):
    git = Path(source) / '.git'
    if git.is_file():
        marker = git.read_text(encoding='utf-8').strip()
        if not marker.startswith('gitdir: '):
            return None
        git = (Path(source) / marker[8:]).resolve()
    if not (git / 'HEAD').is_file():
        return None
    head = (git / 'HEAD').read_text(encoding='utf-8').strip()
    if not head.startswith('ref: '):
        return head
    ref = head[5:]
    if (git / ref).is_file():
        return (git / ref).read_text(encoding='utf-8').strip()
    if (git / 'packed-refs').is_file():
        for line in (git / 'packed-refs').read_text(encoding='utf-8').splitlines():
            if line and not line.startswith(('#', '^')):
                digest, name = line.split(' ', 1)
                if name == ref:
                    return digest
    return None


def runtime_manifest(source, a_root, monitor_file, project_root):
    source, a_root, project_root = Path(source), Path(a_root), Path(project_root)
    model_files = [
        'smpl/envs/atropineenv.py', 'smpl/envs/utils.py',
        'smpl/envs/models/atropine_process.py', 'smpl/envs/models/config.py',
        'smpl/envs/helpers/constants.py', 'smpl/envs/helpers/helper_funcs.py',
        'smpl/configdata/atropineenv/x0.txt', 'smpl/configdata/atropineenv/z0.txt',
        'smpl/configdata/atropineenv/model.npy',
    ]
    return {
        'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
        'dependencies': {p: version(p) for p in ['numpy','scipy','gym','casadi','mzutils','torch']},
        'smpl_source_dir': str(source), 'smpl_checkout_commit': checkout_commit(source),
        'smpl_actual_file_sha256': {name: sha256(source / name) for name in model_files},
        'A_source_dir': str(a_root),
        'A_source_kind': 'actual supplied member_A_11_02 archive; no newer generator source claimed',
        'A_code_sha256': {str(p.relative_to(a_root)): sha256(p) for p in a_root.rglob('*.py')},
        'D_monitor_file': str(Path(monitor_file).resolve()),
        'D_monitor_sha256': sha256(monitor_file),
        'D_safety_rules_sha256': sha256(Path(monitor_file).parent / 'safety_rules.py'),
        'B_code_sha256': {str(p.relative_to(project_root)): sha256(p)
                          for folder in ['integration_abd_1103','data_collection_1104']
                          for p in (project_root / folder).glob('*.py')},
    }
