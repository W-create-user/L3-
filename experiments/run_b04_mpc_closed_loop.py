"""Separate explicitly enabled research MPC result; never the default controller."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.execution.controllers.source_mpc import SourceMPCPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.planning.goal_runtime import run_numeric_goal
from .run_b04_control_study import make_goal

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    summary={'research_only':True,'production_acceptance':False,'success':False,'policy':'source upstream MPC(N=2) plus original buffer P and source capture switch',
             'comparison_scope':'separate hybrid research result; not a pure buffer-MPC-vs-PI optimization comparison'}
    try:
        with (a.output/'runtime_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
            config=json.loads((ROOT/'configs/controllers/mab_buffer_pi.json').read_text());registry=json.loads((ROOT/'configs/environments/mab_physical_registry.json').read_text())
            with MAbAdapter(max_steps=12) as adapter:
                state=adapter._env.steady_observations.copy();state[17]+=.3/adapter._xscale[17]
                adapter.reset(seed=1610,initial_state=state)
                executor=ExecutionAgent(adapter,log_path=a.output/'B_events.jsonl');monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
                system=MultiAgentSystem(executor=executor,monitor=monitor,event_path=a.output/'events.jsonl')
                policy=SourceMPCPolicy(adapter,horizon=2,allow_research=True)
                result=run_numeric_goal(system,make_goal(config,registry,'source_mpc_research'),registry,environment='MAbEnvGym',policy=policy,timeout_seconds=90)
                (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                summary.update(status=result['summary']['status'],reason=result['summary']['reason'],step_calls=adapter.step_calls,
                               initial_state_sha256=hashlib.sha256(state.astype(np.float32).tobytes()).hexdigest(),
                               final_error_dm=result['samples'][-1]['observation'][17]-15,
                               saturation_count=policy.saturation_count,solver_statuses=policy.solver_statuses,
                               all_C_locks_released=not bool(system.scheduler.active_agent_tasks))
                summary['success']=result['summary']['status']=='succeeded' and summary['all_C_locks_released']
    except Exception:summary['traceback']=traceback.format_exc()
    finally:(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return 0 if summary['success'] else 1

if __name__=='__main__':raise SystemExit(main())
