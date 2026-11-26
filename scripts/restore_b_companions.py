"""Restore companion files from the already-public, authorized B25 handoff ZIP.

Copies only configs/dictionary/lock/patches. Never replaces A/B/D Python code,
extracts raw trajectories or downloads new unapproved artifacts. Existing
nonmatching files cause refusal instead of being overwritten.
"""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL=('https://raw.githubusercontent.com/W-create-user/L3-/'
     'ad45d80c002f6cf36f13c592de676a95fa2d6d09/'
     'handoffs/B_to_D_20261125/B_to_D_20261125.zip')
SHA256='e0c0cced1ff3adb94f3e2d6574a4f5607b1f7f6cebc5bc91c9f5fe728a102117'
PREFIX='B_to_D_20261125/04_source_and_companions/'
DIRECTORIES=('configs/controllers/','configs/environments/','configs/scenarios/',
             'configs/training/','patches/')
FILES=('docs/data_dictionary.csv','requirements-runtime-20261120.lock')


def restore(archive,root):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Archive SHA256 differs from public B25 handoff')
    payloads=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith(PREFIX) or name.endswith('/'):continue
            relative=name[len(PREFIX):]
            if not (relative in FILES or relative.startswith(DIRECTORIES)):continue
            path=Path(relative)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid archive path')
            target=root/path
            if not target.resolve().is_relative_to(root.resolve()):raise ValueError('Target leaves checkout')
            data=z.read(name)
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('Refusing to overwrite modified companion: '+relative)
            payloads.append((target,data))
    if not all(any(str(target.relative_to(root))==name for target,_ in payloads) for name in FILES):
        raise ValueError('Archive companion set incomplete')
    for target,data in payloads:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(data)
    print('Verified companion files:',len(payloads),'; no Python source or raw dataset extracted')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    if a.archive:restore(a.archive,a.root)
    else:
        with tempfile.TemporaryDirectory(prefix='b-companions-') as tmp:
            archive=Path(tmp)/'handoff.zip'
            print('Downloading previously authorized public handoff ZIP (about 68 MiB)')
            urllib.request.urlretrieve(URL,archive)
            restore(archive,a.root)


if __name__=='__main__':main()
