"""D independent public acceptance. No private seed access or implied signoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def run(output, integrated=True):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    commands = [(f'D{n:02d}', ['-m','unittest','discover','-s',f'deliverables/D{n:02d}/tests','-v'])
                for n in (1,4,5,6,7)]
    if integrated:
        commands.append(('integrated', ['-m','experiments.verify_a_release','--output',str(output/'integrated')]))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
    for name, args in commands:
        start = time.monotonic()
        with (output/f'{name}.log').open('wb') as log:
            try:
                rc = subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        results.append(dict(case=name,command=[sys.executable,*args],exit_code=rc,wall_seconds=time.monotonic()-start))
        print(name,rc,flush=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0')
    sources = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files if (ROOT/p).is_file()}
    report = dict(source_commit=commit,source_sha256=sources,python=sys.version,cases=results,
                  software_checks_passed=all(x['exit_code']==0 for x in results),
                  formal_holdout_executed=False,independent_human_signoff=None,
                  overall_project_accepted=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    sums = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}' for p in sorted(output.rglob('*')) if p.is_file()]
    (output/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    return report['software_checks_passed']

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--d-only',action='store_true')
    a=p.parse_args()
    sys.exit(0 if run(a.output,not a.d_only) else 1)
