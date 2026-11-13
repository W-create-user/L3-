"""Independent real-simulator episodes with explicitly scoped injected sensor labels.
No physical process health labels are inferred. Does not enable an R-03 rule.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from l3_system.execution.environment_adapter import EnvironmentAdapter
from l3_system.execution.execution_agent import ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'label_scope':'known injected observation corruption only; not physical process health',
              'physical_process_labels_available':False,'physical_process_iforest_ready':False,
              'sensor_fault_benchmark_available':True,'horizon_steps':30,'sampling_period_min':10,
              'source_base_commit':'c3dc5cc12d5f5bdd2ae39673bcf74529a24912d3',
              'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'episodes':[]}
    def save():
        (args.output/'episodes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    save()
    for split,seeds in [('train',range(1301,1307)),('validation',range(1401,1404)),('test',range(1501,1504))]:
        for seed in seeds:
            ident=f'{split}_{seed}';folder=args.output/ident;folder.mkdir()
            episode={'episode_id':ident,'group_id':f'initial_condition_seed_{seed}',
                     'split':split,'seed':seed,'reset_count':1,'requested_steps':30,
                     'complete':False,'process_disturbance':None,'process_health_label':None,
                     'fault':None if split=='train' else {'target':'delivered_observation[6]','type':'multiplicative_sensor_bias','multiplier':1.2,'first_step':10,'last_step':20,'phase':'after real env.step; does not change environment state or action','basis':'B-designed synthetic sensor-fault benchmark; not A-approved physical anomaly'},
                     'label_definition':{'0':'no synthetic sensor corruption applied to this observation','1':'the declared synthetic sensor corruption was applied; independent of threshold or detector score'},
                     'raw_trajectory':ident+'/trajectory.jsonl'}
            manifest['episodes'].append(episode);save()
            try:
                with EnvironmentAdapter(max_steps=60) as env:
                    executor=ExecutionAgent(env,log_path=root/'work/r03_private_logs'/f'{ident}.jsonl')
                    observation=executor.reset(seed=seed)
                    initial=np.asarray(env._env.xk.full(),dtype=np.float64)
                    episode['initial_internal_state_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
                    episode['initial_observation_sha256']=hashlib.sha256(np.asarray(observation,dtype=np.float64).tobytes()).hexdigest()
                    mon=MonitoringAgent(env.action_space,env.observation_space)
                    with (folder/'trajectory.jsonl').open('w',encoding='utf-8') as f:
                        f.write(json.dumps({'episode_id':ident,'step':0,'sim_time_min':0,'kind':'reset','environment_observation':observation,'delivered_observation':observation,'sensor_fault_label':0,'process_health_label':None})+'\n')
                        for step in range(1,31):
                            action=np.zeros(env.action_space.shape,dtype=env.action_space.dtype)
                            request=executor.prepare_action({'task_id':f'{ident}_{step}','task_type':'control_cycle'},action=action)
                            with contextlib.redirect_stdout(io.StringIO()):
                                review=mon.check_request(request,request['observation'])
                            if review['decision']!='pass':raise RuntimeError('Unexpected request rejection')
                            result=executor.execute_action(request,review)
                            if not result['success'] or not result['executed']:raise RuntimeError('Failed environment step')
                            raw=list(result['observation']);delivered=list(raw)
                            active=split!='train' and 10<=step<=20
                            if active:delivered[6]=float(raw[6]*1.2)
                            row={'episode_id':ident,'step':step,'sim_time_min':step*10,
                                 'requested_action':request['action'],'actual_action':result['action'],
                                 'environment_observation':raw,'delivered_observation':delivered,
                                 'raw_preclip_observation':np.asarray(env._env.previous_observation).tolist(),
                                 'reward':result['reward'],'done':result['done'],
                                 'terminated':result['terminated'],'truncated':result['truncated'],
                                 'sensor_fault_label':int(active),'process_health_label':None,
                                 'label_source':'declared_injection_schedule_not_threshold_or_reward',
                                 'request_id':result['request_id'],'source_step_count':result['step_count']}
                            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                            episode['recorded_steps']=step
                            if result['done']:raise RuntimeError('Unexpected early terminal: partial episode preserved')
                    episode['complete']=True;episode['stop_reason']='declared_30_step_budget'
                    episode['trajectory_sha256']=hashlib.sha256((folder/'trajectory.jsonl').read_bytes()).hexdigest()
            except Exception:
                import traceback
                episode['error']=traceback.format_exc();save();raise
            save();print(ident,'complete',flush=True)
    # Independence means separate reset/process instances, not statistical independence.
    digests=[e['initial_internal_state_sha256'] for e in manifest['episodes']]
    if len(set(digests))!=len(digests):
        raise RuntimeError('Duplicate initial states found: regroup before using train/test split')
    manifest['unique_initial_state_hashes']=len(set(digests));save()


if __name__=='__main__':main()
