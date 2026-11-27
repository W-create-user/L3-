"""Explicit live no-step/stale-request/error-reward checks; faults stay offline."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from l3_system.execution import MAbAdapter, ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);checks={}
    with MAbAdapter(max_steps=1) as a:
        e=ExecutionAgent(a,log_path=args.output/'B_events.jsonl');e.reset(seed=1610)
        m=MonitoringAgent(a.action_space,a.observation_space)
        before=(a.state_version,a.step_calls,a.step_count)
        for action in [np.zeros(8),np.full(9,np.nan),a.action_space.high.astype(float)+1]:
            try:a.step(action)
            except ValueError:pass
            else:raise RuntimeError('Invalid action was executed')
        if (a.state_version,a.step_calls,a.step_count)!=before:raise RuntimeError('Invalid action mutated environment')
        checks['invalid_action_did_not_advance']=True
        req=e.prepare_action({'task_id':'stale','task_type':'control_cycle'},action=a._env.steady_actions)
        with contextlib.redirect_stdout(io.StringIO()):review=m.check_request(req,req['observation'])
        old_episode=a.metadata['episode_id'];e.reset(seed=1611)
        if a.metadata['episode_id']==old_episode:raise RuntimeError('Episode identity did not change')
        try:e.execute_action(req,review)
        except (ValueError,RuntimeError):pass
        else:raise RuntimeError('Stale request was accepted')
        if a.step_calls!=0:raise RuntimeError('Stale request advanced environment')
        checks['reset_invalidates_request']=True
        def fail(*unused):raise RuntimeError('B03 offline injected solver failure')
        a._env._simulation=fail
        with contextlib.redirect_stdout(io.StringIO()) as captured:result=a.step(a._env.steady_actions)
        if result['success'] or result['reward']!=-100 or not a.done:raise RuntimeError('Error reward mistaken for success')
        try:a.require_ready()
        except RuntimeError:pass
        else:raise RuntimeError('Failed environment still ready')
        checks['failed_solver_is_not_success']=True
        (args.output/'injected_failure.json').write_text(json.dumps({'result':result,'model_stdout':captured.getvalue()},indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps({'success':True,'checks':checks,'fault_scope':'offline test only'},indent=2)+'\n')

if __name__=='__main__':main()
