"""Build an isolated public A08/B component review copy, without private data.

Run from a full repository clone containing the pinned commits. Does not switch
or edit the caller's checkout. The default boundary ref is the caller's HEAD.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

A_REF='08ea1796b6ecdc6d20f2f569f502e9f4e8edc351'
B_REF='68831ef9e06440a1066e9d18bfb3955e1ac3006f'
B_BASE='4a09dcd'
BOUNDARY_FILES=('l3_system/execution/controllers/receipt_boundary.py',
                'l3_system/tests/test_receipt_boundary.py','experiments/verify_b_freeze_receipts.py')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--boundary-ref',default='HEAD');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
    boundary=git('rev-parse',a.boundary_ref).decode().strip()
    # Validate every source/ref before creating the output directory.
    archive=git('archive',A_REF)
    paths=git('diff','--name-only',B_BASE,B_REF).decode().splitlines()
    if any(Path(name).suffix not in ('.py','.sh','.ps1') for name in paths):raise ValueError('Unexpected non-code B overlay')
    payloads={name:git('show',B_REF+':'+name) for name in paths}
    payloads.update({name:git('show',boundary+':'+name) for name in BOUNDARY_FILES})
    a.destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(a.destination,filter='data')
    for name,data in payloads.items():
        target=a.destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    plan=json.loads((a.destination/'configs/research/a08_mab_joint_freeze_candidate_20261126.json').read_text())
    for group in ('A_C_source_sha256','B_source_sha256'):
        for name,expected in plan['components'][group].items():
            if hashlib.sha256((a.destination/name).read_bytes()).hexdigest()!=expected:raise ValueError('Component mismatch: '+name)
    evidence={'A_tree':A_REF,'B_components':B_REF,'boundary_commit':boundary,'component_hashes_match':True,
        'boundary_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in payloads.items() if name in BOUNDARY_FILES},
        'holdout_read':False,'formal_launch_authorized':False}
    (a.destination/'joint_review_sources.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Public review copy ready:',a.destination.resolve())
    print('Enter that directory; run python -m unittest l3_system.tests.test_acknowledged_buffer l3_system.tests.test_receipt_boundary -v')
    print('Then python -m experiments.verify_b_freeze_receipts --output <new-output-directory>')


if __name__=='__main__':main()
