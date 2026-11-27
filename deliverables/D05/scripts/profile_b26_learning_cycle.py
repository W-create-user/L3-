"""B26+D same-host full-cycle development profile, old seeds only."""
import argparse,contextlib,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.execution import MAbAdapter,ExecutionAgent
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.runtime import MultiAgentSystem
from l3_system.monitoring.b26_learning_gate import B26LearningGate,FailClosedPolicy,validate_anchor
from deliverables.D07.scripts.run_mab_paired import definitions,make_plan
from experiments.run_b08_development import RecordedAdapter,MeasuredPolicy

def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 config=json.loads(a.config.read_text());configsha=hashlib.sha256(a.config.read_bytes()).hexdigest();rows=[]
 for condition,seed,offset in [('nominal',26001,.02),('sensor_fault',26011,-.02),('process_disturbance',26021,.2)]:
  folder=a.output/condition;folder.mkdir();gate=None;cycles=[];start=time.monotonic();failure=None
  try:
   with (folder/'stdout.log').open('w') as log,contextlib.redirect_stdout(log),MAbAdapter(max_steps=10) as base:
    initial=base._env.steady_observations.copy();initial[17]+=offset/base._xscale[17];base.reset(seed=seed,initial_state=initial)
    adapter=RecordedAdapter(base,folder);ex=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl');_,_,rules=definitions(adapter,{'run_id':condition})
    context={'environment':'mAb','episode_id':base.metadata['episode_id'],'variables':{'h_model':{'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}}}
    hard=MonitoringAgent(adapter.action_space,adapter.observation_space,process_rules=ProcessRuleEngine(rules),process_context_provider=lambda *_:context)
    gate=B26LearningGate(adapter,hard,a.model,config['model_sha256'],a.config,configsha,source_root=Path(__file__).resolve().parents[3],development_opt_in=True,resources_clear=lambda:not bool(ex._pending) and not bool(system.scheduler.active_agent_tasks))
    certificate={'anchor_id':condition+'-pre-fault','kind':'trusted_simulation_reset_before_fault','episode_id':base.metadata['episode_id'],'state_version':base.state_version,'level_dm':float(base.get_observation()[17]*base._xscale[17]),'unit':'dm','coordinate':'physical_projection','source_ref':'B68831ef+source reset reference; no field calibration claim','observed_monotonic':time.monotonic(),'fault_started':False}
    level=validate_anchor(certificate,adapter);write(folder/'startup_certificate.json',certificate)
    policy=FailClosedPolicy(AcknowledgedBufferPolicy(adapter,make_plan()['controller'],startup_level_dm=level,startup_trusted=True,episode_id=base.metadata['episode_id'],state_version=base.state_version,allow_research=True))
    adapter.policy=policy;measurements=[];view=MeasuredPolicy(policy,adapter,{'condition':condition},measurements)
    system=MultiAgentSystem(executor=ex,monitor=hard,event_path=folder/'A_C_events.jsonl')
    for index in range(10):
     if index==6:
      gate.confirm('D26-explicit-development-review-after-six-real-samples');system.monitor=gate
     before=time.perf_counter();old_count=base.step_calls
     result=system.run_cycle(action_generator=lambda obs,space,attempt:view.predict(obs,space))
     ok=result['summary']['run_status']=='completed';policy.commit(ok)
     if not ok:raise RuntimeError('cycle did not complete: '+str(result['summary']))
     gate.observe_executed(adapter.traces[-1])
     elapsed=time.perf_counter()-before
     cycles.append({'index':index+1,'mode':'explicit_restricted_sampling' if index<6 else 'learned_gate_cycle','whole_cycle_seconds':elapsed,'step_delta':base.step_calls-old_count,'all_C_locks_released':not bool(system.scheduler.active_agent_tasks),'pending_clear':not bool(ex._pending),'state_version':base.state_version})
     write(folder/'cycles.json',cycles)
     if gate.failed:raise RuntimeError('post-step learned review stopped session')
    write(folder/'observer_trace.json',policy.events);write(folder/'measurements.json',measurements);write(folder/'D_gate_events.json',gate.events)
  except Exception as exc:failure=type(exc).__name__+': '+str(exc)
  finally:
   if gate is not None:
    write(folder/'D_gate_events.json',gate.events);gate.close()
   rows.append({'condition':condition,'seed':seed,'complete':failure is None,'failure':failure,'wall_seconds':time.monotonic()-start,'cycles':cycles});write(a.output/'summary.json',{'scope':'same-host B+A/C+D process scorer, old-seed development; not goal efficiency or formal holdout','config_sha256':configsha,'B_commit':config['B_commit'],'model_sha256':config['model_sha256'],'results':rows,'joint_freeze_approved':False});print(condition,len(cycles),failure,flush=True)
 if any(not r['complete'] for r in rows):return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
