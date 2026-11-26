"""Bounded source MPC probe in a child process; never executes a plant action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def worker(path):
    report={'source_helper':'vendor/smpl/smpl/envs/helpers/mab_helpers.py:_build_mpc_up',
            'horizon':2,'plant_step_calls':0,'controller_enabled':False,'success':False}
    started=time.perf_counter()
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(25,25))
        import numpy as np
        from l3_system.execution import MAbAdapter
        with MAbAdapter(max_steps=2) as adapter:
            c=adapter._env.controller
            solver=c._build_mpc_up(np.eye(17),np.eye(7),2,c.dt_spl,c.xss[:17],c.uss[:7],c.xscale[:17],c.uscale[:7])
            solver.initialize(casadioptions={'ipopt.linear_solver':'mumps','ipopt.print_level':0},solveroptions={'tol':1e-5})
            solver.fixvar('x',0,c.xss[:17]/c.xscale[:17]);solver.solve()
            report['solver_status']=str(solver.stats.get('status'))
            action=np.squeeze(solver.var['u',0]);report['requested_upstream_action']=action.tolist()
            if report['solver_status']!='Solve_Succeeded' or action.shape!=(7,) or not np.isfinite(action).all():
                raise RuntimeError('MPC did not produce a verified finite successful solution')
            report['success']=True
    except Exception:
        report['traceback']=traceback.format_exc()
    finally:
        report['wall_seconds']=time.perf_counter()-started
        path.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['success'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:return worker(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-m','experiments.probe_mpc_feasibility','--worker','--output',str(a.output/'worker_report.json')]
    report={'command':command,'controller_enabled':False,'scope':'two-step source solver feasibility only, not closed-loop MPC acceptance'}
    with (a.output/'worker.log').open('w') as log:
        try:
            run=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=35)
            report['exit_code']=run.returncode
        except subprocess.TimeoutExpired:report['timeout_seconds']=35
    if (a.output/'worker_report.json').exists():report['worker_report']=json.loads((a.output/'worker_report.json').read_text())
    report['feasible_short_solver_probe']=bool(report.get('exit_code')==0 and report.get('worker_report',{}).get('success'))
    report['source_sha256']=hashlib.sha256((ROOT/'vendor/smpl/smpl/envs/helpers/mab_helpers.py').read_bytes()).hexdigest()
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='worker_report'},indent=2))
    # A blocked optional MPC probe is a reported outcome, not an enabled controller.
    return 0

if __name__=='__main__':raise SystemExit(main())
