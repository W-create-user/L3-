"""B08 development-only paired control attribution using disclosed old D seeds.

Not a blind holdout launch. Requires explicit trusted pre-fault startup opt-in.
The new controller sees only startup calibration, delivered h and acknowledged
executed actions; the test harness alone injects bias and scores true states.
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
from l3_system.execution import MAbAdapter, ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID
from l3_system.execution.controllers.acknowledged_buffer import AcknowledgedBufferPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.process_rules import ProcessRuleEngine
from l3_system.runtime import MultiAgentSystem
from l3_system.planning.goal_runtime import run_numeric_goal, GoalCoordinates
from l3_system.planning.goals import evaluate_goal
from deliverables.D07.scripts import run_mab_paired as d24

ROOT=Path(__file__).resolve().parents[1]
ARMS=('baseline_P','research_PI','observer_direct','observer_four')


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class MeasuredPolicy:
    def __init__(self,base,adapter,spec,events):
        self.base=base;self.adapter=adapter;self.spec=spec;self.events=events
    def reset(self):self.base.reset()
    def predict(self,observation,space):
        # Only the harness knows this predefined development scenario.
        delivered=np.asarray(observation,dtype=float).copy()
        bias=.10 if self.spec['condition']=='sensor_fault' and self.adapter.step_count<3 else 0.
        delivered[17]+=bias/self.adapter._xscale[17]
        self.events.append({'step_before':self.adapter.step_count,'injected_bias_dm':bias,
                            'delivered_h_dm':float(delivered[17]*self.adapter._xscale[17])})
        return self.base.predict(delivered,space)
    def commit(self,executed):self.base.commit(executed)


class RecordedAdapter:
    def __init__(self,base,folder):
        self.base=base;self.folder=folder;self.policy=None;self.traces=[]
    def __getattr__(self,key):return getattr(self.base,key)
    def step(self,action):
        result=self.base.step(action)
        self.traces.append(result)
        # Persist before acknowledgement; errors after physical execution survive.
        with (self.folder/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(result,allow_nan=False)+'\n')
        if hasattr(self.policy,'acknowledge'):
            self.policy.acknowledge(result)
        return result


def run_one(spec,plan,output):
    folder=output/spec['run_id'];folder.mkdir()
    row={**spec,'started':False,'success':False,'status':'initializing','failure_reason':None,
         'step_calls':0,'successful_step_calls':0,'simulated_makespan_min':None,
         'wall_seconds':None,'startup_calibration':'trusted simulation reset before controller-only corruption'}
    started=time.monotonic();adapter=None;policy=None;base=None;monitor=None;ex=None
    try:
        with (folder/'stdout.log').open('w') as log,contextlib.redirect_stdout(log),MAbAdapter(max_steps=10) as base:
            initial=base._env.steady_observations.copy()
            initial[17]+=spec['initial_buffer_offset_dm']/base._xscale[17]
            base.reset(seed=spec['seed'],initial_state=initial);row['started']=True
            row['initial_internal_state_sha256']=hashlib.sha256(np.asarray(base._env.Xi[0],dtype=np.float64).tobytes()).hexdigest()
            adapter=RecordedAdapter(base,folder)
            ex=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl')
            registry,goal,rules=d24.definitions(adapter,spec)
            context={'environment':'mAb','episode_id':base.metadata['episode_id'],
                     'variables':{'h_model':{'unit':'dm / xscale[17]','coordinate':'physical_divided_by_xscale'}}}
            monitor=MonitoringAgent(adapter.action_space,adapter.observation_space,
                process_rules=ProcessRuleEngine(rules),process_context_provider=lambda *_:context)
            config=plan['PI_config'] if spec['arm']=='research_PI' else plan['controller']
            if spec['arm'].startswith('observer'):
                policy=AcknowledgedBufferPolicy(adapter,config,
                    startup_level_dm=float(base.get_observation()[17]*base._xscale[17]),
                    startup_trusted=True,episode_id=base.metadata['episode_id'],state_version=base.state_version,
                    allow_research=True,max_innovation_dm=.3)
            else:policy=BufferLevelPID(adapter,config)
            adapter.policy=policy;events=[];view=MeasuredPolicy(policy,adapter,spec,events)
            if spec['arm']=='observer_four':
                system=MultiAgentSystem(executor=ex,monitor=monitor,event_path=folder/'A_C_events.jsonl')
                remaining=90-(time.monotonic()-started)
                if remaining<=0:raise TimeoutError('budget exhausted during initialization')
                result=run_numeric_goal(system,goal,registry,environment='MAbEnvGym',policy=view,timeout_seconds=remaining)
                write(folder/'goal_result.json',result)
                row.update(success=result['summary']['status']=='succeeded',status=result['summary']['status'],
                           failure_reason=result['summary']['reason'],C_resources_clear=not bool(system.scheduler.active_agent_tasks))
            else:
                view.reset();coordinates=GoalCoordinates(adapter,registry);samples=[]
                def sample():
                    return {'goal_id':goal['goal_id'],'environment_id':'MAbEnvGym','registry_version':registry['registry_version'],
                        'episode_id':base.metadata['episode_id'],'step':base.step_count,
                        'observation':coordinates.observation(ex.get_observation()),
                        'safety_ok':monitor.check_observation(ex.get_observation())['approved']}
                samples.append(sample())
                while True:
                    evaluation=evaluate_goal(samples[-1],samples[:-1],goal,registry)
                    if evaluation['status']!='pending':break
                    if time.monotonic()-started>=90:
                        evaluation.update(status='failed',reason='wall_deadline');break
                    action=view.predict(ex.get_observation(),adapter.action_space)
                    result=ex.execute_assignment({'task_id':f"{spec['run_id']}-{base.step_count}",
                        'task_type':'control_cycle'},monitor.check_request,action=action)
                    executed=result.get('success') is True and result.get('executed') is True
                    view.commit(executed)
                    if not executed:
                        evaluation.update(status='failed',reason=result.get('failure_reason'));break
                    samples.append(sample())
                write(folder/'goal_result.json',{'evaluation':evaluation,'samples':samples})
                row.update(success=evaluation['status']=='succeeded',status=evaluation['status'],failure_reason=evaluation['reason'])
            write(folder/'measurement_events.json',events)
            row['pending_requests_clear']=not bool(ex._pending)
            if row['success']:
                row['simulated_makespan_min']=base.step_count*60
                if time.monotonic()-started>90:
                    row.update(success=False,status='failed',failure_reason='whole_run_wall_deadline',simulated_makespan_min=None)
    except Exception:
        row.update(success=False,status='exception',failure_reason=traceback.format_exc(),simulated_makespan_min=None)
    finally:
        if adapter is not None:
            row['step_calls']=adapter.step_calls
            row['successful_step_calls']=sum(r.get('success') is True for r in adapter.traces)
            row['model_clipping_indices']=sorted({i for r in adapter.traces for i in r.get('info',{}).get('clipped_indices',[])})
            row['buffer_clip_calls']=sum(17 in r.get('info',{}).get('clipped_indices',[]) for r in adapter.traces)
        if hasattr(policy,'events'):write(folder/'observer_trace.json',policy.events)
        if monitor is not None:monitor.export_logs(folder/'D_reviews.jsonl')
        row['wall_seconds']=time.monotonic()-started
        write(folder/'run_summary.json',row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trusted-startup-development',action='store_true');p.add_argument('--pilot',action='store_true')
    p.add_argument('--pi-config',type=Path,default=ROOT/'configs/controllers/mab_buffer_pi.json')
    a=p.parse_args()
    if not a.trusted_startup_development:p.error('explicit trusted startup assumption required; not formal acceptance')
    a.output.mkdir(parents=True,exist_ok=False)
    plan=d24.make_plan();original=plan['runs']
    specs=[r for r in original if r['arm']=='traditional_controller']
    if a.pilot:specs=[next(r for r in specs if r['condition']==kind) for kind in ('nominal','sensor_fault','process_disturbance')]
    runs=[]
    for spec in specs:
        for arm in ARMS:runs.append({**spec,'arm':arm,'run_id':spec['pair_id']+'-'+arm})
    random.Random(261126).shuffle(runs)
    plan.update(schema_version='B08Development/1.0',date='2026-11-26',formal_blind_test=False,
        joint_freeze_confirmed=False,pilot=a.pilot,runs=runs,arms={a:a for a in ARMS},
        PI_config=json.loads(a.pi_config.read_text()),
        trusted_startup_required=True,calibration='pre-fault simulator reset measurement, explicit assumption',
        unsupported=['arbitrary initially biased calibration','unmeasured flow disturbances','industrial safety certification'],
        old_seed_development_only=True,controller_truth_access='initial trusted calibration only; thereafter delivered h and acknowledged executed actions',
        efficiency_scope='control attribution; same P+observer direct vs A/C one executor; not distributed scheduling gain')
    for path in ['experiments/run_b08_development.py','l3_system/execution/controllers/acknowledged_buffer.py']:
        plan['source_sha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    for top in ['l3_system','vendor/smpl']:
        for path in sorted((ROOT/top).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.npy') and 'tests' not in path.parts:
                plan['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    plan['dependencies']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    write(a.output/'frozen_plan.json',plan);rows=[]
    for i,spec in enumerate(runs,1):
        row=run_one(spec,plan,a.output);rows.append(row);write(a.output/'run_ledger.json',rows)
        print(i,len(runs),spec['run_id'],row['status'],row['step_calls'],flush=True)
    write(a.output/'completion.json',{'planned':len(runs),'receipts':len(rows),'pilot':a.pilot,
        'all_receipts_retained':True,'formal_blind_test':False,'physical_safety_certified':False})
    return 0 if all(r['started'] and r['success'] for r in rows) else 1


if __name__=='__main__':raise SystemExit(main())
