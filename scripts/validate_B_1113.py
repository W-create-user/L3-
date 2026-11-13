"""Validate B02 daily evidence without a simulator installation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'deliveries/B_20261113'

def read(name):
    return json.loads((PACK/name).read_text(encoding='utf-8'))

def rows(name):
    return [json.loads(s) for s in (PACK/'runs'/name/'trajectory.jsonl').read_text().splitlines()]

def check(condition,message):
    if not condition:
        raise ValueError(message)

def main():
    manifest=read('manifest.json')['files']
    for name,expected in manifest.items():
        check(hashlib.sha256((PACK/name).read_bytes()).hexdigest()==expected,f'Hash mismatch: {name}')
    tools={hashlib.sha256((ROOT/'experiments/verify_mab_daily.py').read_bytes()).hexdigest(),hashlib.sha256((PACK/'tool_versions/initial_probe.py').read_bytes()).hexdigest()}
    for name in ['seed1113','seed1113_replay','seed1114','termination','injected_solver_failure','profiled_seed1113']:
        report=read(f'runs/{name}/report.json')
        check(report['tool_sha256'] in tools,'Tool source hash mismatch')
        check(report['spaces']=={'observation':[1970],'action':[9]},'Live spaces mismatch')
        for row in rows(name):
            check(len(row['observation'])==1970,'Trajectory shape mismatch')
            if row['kind']=='step':
                check(len(row['requested_action'])==9,'Action dimension mismatch')
                check(row['sim_time_min']==row['step_count']*60,'Simulation time mismatch')
        check(report['success']==(name!='injected_solver_failure'),'Failure reported as success')
    a,b,c=rows('seed1113'),rows('seed1113_replay'),rows('seed1114')
    check(a[0]['observation']==b[0]['observation'] and a[1]['observation']==b[1]['observation'],'Same-seed replay differs')
    check(a[0]['observation']!=c[0]['observation'],'Different seeds did not change initial state')
    failure=read('runs/injected_solver_failure/report.json')
    check(failure['successful_steps']==0 and rows('injected_solver_failure')[1]['reward']==-100,'Error reward not detected')
    terminal=read('runs/termination/report.json')
    check(terminal['successful_steps']==2 and terminal['last_info']['timeout'],'Termination boundary changed')
    check('Ran 90 tests' in (PACK/'unit_tests_final.log').read_text(),'Unit test evidence missing')
    print(f'Validated {len(manifest)} SHA256 hashes, real steps, deterministic replay, seed variation, expected solver failure and termination boundary.')

if __name__=='__main__':
    main()
