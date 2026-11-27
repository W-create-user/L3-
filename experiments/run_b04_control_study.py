"""Source PI, steady and midpoint baselines through the actual A/B/C/D goal chain."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from l3_system.execution import MAbAdapter, ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID, SpaceMidpointPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.planning.goal_runtime import run_numeric_goal

ROOT=Path(__file__).resolve().parents[1]

def make_goal(config,registry,name):
    return {'schema_version':'2.0-draft','goal_id':name,'environment_id':'MAbEnvGym','registry_version':registry['registry_version'],
            'variable':'buffer_height','unit':'dm','coordinate':'physical_projection',
            'target_range':{'lower':config['setpoint_dm']-config['target_tolerance_dm'],'upper':config['setpoint_dm']+config['target_tolerance_dm']},
            'max_steps':config['horizon_steps'],'hold_steps':config['hold_steps'],
            'allowed_actions':{n:{'unit':v['unit'],'coordinate':v['coordinate'],'range':v['bounds']} for n,v in registry['variables'].items() if v['source']=='action'}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/controllers/mab_buffer_pi.json').read_text());registry=json.loads((ROOT/'configs/environments/mab_physical_registry.json').read_text())
    cases=[('PI_train',1610,.3,'PI'),('PI_holdout_positive',1611,.15,'PI'),('PI_holdout_negative',1612,-.2,'PI'),
           ('source_steady',1610,.3,'steady'),('space_midpoint',1610,.3,'midpoint')]
    summary={'research_only':True,'production_acceptance':False,'control_config_sha256':hashlib.sha256((ROOT/'configs/controllers/mab_buffer_pi.json').read_bytes()).hexdigest(),
             'MPC_enabled':False,'MPC_scope':'separate source solver feasibility probe; no full closed-loop MPC acceptance','cases':[],'success':False}
    traces=[]
    try:
        with (a.output/'runtime_stdout.log').open('w') as log,contextlib.redirect_stdout(log):
            for name,seed,offset,kind in cases:
                folder=a.output/name;folder.mkdir()
                with MAbAdapter(max_steps=12) as adapter:
                    state=adapter._env.steady_observations.copy();state[17]+=offset/adapter._xscale[17]
                    adapter.reset(seed=seed,initial_state=state)
                    initial_hash=hashlib.sha256(state.astype(np.float32).tobytes()).hexdigest()
                    executor=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl');monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
                    system=MultiAgentSystem(executor=executor,monitor=monitor,event_path=folder/'events.jsonl')
                    policy=BufferLevelPID(adapter,config) if kind=='PI' else SpaceMidpointPolicy() if kind=='midpoint' else (lambda *_:adapter._env.steady_actions.copy())
                    result=run_numeric_goal(system,make_goal(config,registry,name),registry,environment='MAbEnvGym',timeout_seconds=90,policy=policy)
                    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                    errors=[sample['observation'][17]-config['setpoint_dm'] for sample in result['samples']]
                    actions=[];valid_steps=0
                    for cycle in result['cycles']:
                        for stage in cycle['stage_results']:
                            if stage['task_type']=='execute_approved_action' and 'execution_result' in stage:
                                execution=stage['execution_result'];valid_steps+=int(execution.get('success') is True and execution.get('executed') is True)
                                if execution.get('action') is not None:actions.append(adapter.action_coordinates.to_physical(execution['action']).tolist())
                    metrics={'name':name,'policy':kind,'seed':seed,'initial_buffer_offset_dm':offset,'initial_state_sha256':initial_hash,
                             'goal_achieved':result['summary']['status']=='succeeded','status':result['summary']['status'],
                             'reason':result['summary']['reason'],'step_calls':adapter.step_calls,'successful_steps':valid_steps,
                             'final_error_dm':errors[-1],'sum_poststep_abs_error_dm_min':sum(abs(e) for e in errors[1:])*60,
                             'saturation_count':policy.saturation_count if kind=='PI' else None,
                             'all_C_locks_released':not bool(system.scheduler.active_agent_tasks),
                             'requested_and_executed_actions_path':name+'/result.json'}
                    summary['cases'].append(metrics);traces.append((name,errors,actions,result['summary']['status']))
                    if kind=='PI' and not metrics['goal_achieved']:raise RuntimeError('Frozen PI did not achieve research goal')
                    if not metrics['all_C_locks_released']:raise RuntimeError('C lock leaked')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,(level,flow)=plt.subplots(2,1,figsize=(8,6))
        for name,errors,actions,status in traces:
            level.plot(np.arange(len(errors))*60,np.asarray(errors)+config['setpoint_dm'],marker='.',label=name+': '+status)
            if actions:flow.step(np.arange(1,len(actions)+1)*60,[row[7] for row in actions],where='post',marker='.',label=name+': '+status)
        level.axhspan(14.95,15.05,color='green',alpha=.12);level.set_ylabel('buffer height (dm)');level.legend(fontsize=7)
        flow.set_ylabel('requested outlet (L/min)');flow.set_xlabel('simulation time (min)');flow.legend(fontsize=7)
        fig.tight_layout();fig.savefig(a.output/'response_curves.png',dpi=120);plt.close(fig)
        summary['success']=True
    except Exception:summary['traceback']=traceback.format_exc();raise
    finally:(a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
