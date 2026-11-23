"""D-owned fresh simulator benchmark using unchanged B execution interfaces.

Artificial sensor bias is applied only to the offline delivered copy. Physical
health remains unknown. The declared next-benchmark schedule is fixed before
collection and the single model/threshold recipe is not selected using test.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(plan_path, output):
    from deliverables.D01.scripts.check_next_sensor_plan import validate
    from l3_system.execution import EnvironmentAdapter, ExecutionAgent
    from l3_system.agents.monitoring_agent import MonitoringAgent
    plan=json.loads(Path(plan_path).read_text());validate(plan)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'collection_plan.json').write_bytes(Path(plan_path).read_bytes())
    root=Path(__file__).resolve().parents[2]
    manifest={'schema_version':'DNextSensorCollection/1.0','complete':False,
        'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':digest(plan_path),
        'authorization':'User request 2026-11-23 to complete all due D tasks; new isolated simulation evidence only',
        'original_plan_status':'draft retained unchanged; this run is a separate D collection, not a team formal experiment freeze',
        'recipe':{'window_size':4,'seed':1201,'trees':100,'threshold':'validation maximum F1, ties higher threshold',
                  'normalization':'train only','observation_0':'replaced with zero before model features',
                  'test_policy':'single evaluation after fixed training and validation recipe; no tuning on test'},
        'physical_health_labels_available':False,'deployment_authorized':False,'episodes':[],
        'source_sha256':{n:digest(root/n) for n in ['l3_system/monitoring/sensor_benchmark.py',
            'l3_system/monitoring/anomaly_detector.py','l3_system/execution/environment_adapter.py',
            'l3_system/execution/execution_agent.py','vendor/smpl/smpl/envs/atropineenv.py']}}
    def save(): (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    save();seen={}
    try:
        for spec in plan['episodes']:
            eid=spec['episode_id'];folder=output/eid;folder.mkdir()
            row={'episode_id':eid,'split':spec['split'],'lineage_group':spec['group_id'],
                 'label_scope':'sensor_fault','label_source':'fixed artificial injection schedule, not model outputs',
                 'observations':[],'labels':[],'complete':False}
            with (folder/'stdout.txt').open('w') as log,redirect_stdout(log), EnvironmentAdapter(max_steps=65) as adapter:
                executor=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl');executor.reset(seed=spec['seed'])
                initial=hashlib.sha256(np.asarray(adapter._env.xk.full(),dtype=np.float64).tobytes()).hexdigest()
                if initial in seen:raise ValueError('duplicate initial state source across scheduled episodes')
                seen[initial]=eid;row['initial_internal_state_sha256']=initial
                monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
                with (folder/'trajectory.jsonl').open('w') as stream:
                    for step in range(1,spec['horizon_steps']+1):
                        action=(np.asarray(adapter.action_space.low)+np.asarray(adapter.action_space.high))/2
                        result=executor.execute_assignment({'task_id':f'{eid}-{step}','task_type':'control_cycle'},
                            monitor.check_request,action=action)
                        if not result['success'] or not result['executed'] or result['done']:
                            raise RuntimeError('failed or terminal step; partial evidence retained')
                        raw=list(result['observation']);delivered=list(raw);fault=spec['fault']
                        active=bool(fault and fault['first_step']<=step<=fault['last_step'])
                        if active:delivered[6]*=fault['multiplier']
                        features=list(delivered);features[0]=0.
                        row['observations'].append(features);row['labels'].append(active)
                        record={'step':step,'source_state_version':adapter.state_version,'simulation_time_min':step*10,
                                'environment_observation':raw,'delivered_observation':delivered,
                                'sensor_fault_label':active,'physical_health_label':None,
                                'executed_action':result['action'],'request_id':result['request_id']}
                        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                row['complete']=True
                (folder/'model_episode.json').write_text(json.dumps(row,allow_nan=False)+'\n')
                manifest['episodes'].append({'episode_id':eid,'split':spec['split'],'seed':spec['seed'],
                    'initial_internal_state_sha256':initial,'steps':len(row['observations']),
                    'positive_sensor_records':sum(row['labels']),
                    'episode_path':eid+'/model_episode.json','episode_sha256':digest(folder/'model_episode.json'),
                    'trajectory_sha256':digest(folder/'trajectory.jsonl')})
            save();print(eid,'complete',flush=True)
        manifest['complete']=True;save()
    except Exception:
        manifest['failure_traceback']=traceback.format_exc();save();raise
    return manifest


def evaluate(output):
    from .anomaly_detector import AnomalyDetector
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('partial collection cannot be evaluated')
    evaluation=output/'evaluation';evaluation.mkdir(exist_ok=False)
    episodes=[]
    for entry in manifest['episodes']:
        path=output/entry['episode_path']
        if digest(path)!=entry['episode_sha256']:raise ValueError('episode bytes changed')
        episodes.append(json.loads(path.read_text()))
    detector=AnomalyDetector(window_size=4,seed=1201)
    card=detector.fit(episodes,label_scope='sensor_fault')
    detector.save(evaluation/'model')
    summary={'schema_version':'DNextSensorResult/1.0','collection_manifest_sha256':digest(output/'manifest.json'),
             'episode_counts':{s:sum(e['split']==s for e in episodes) for s in ('train','validation','test')},
             'total_steps':sum(len(e['observations']) for e in episodes),'model':card,
             'whole_normal':{},'probability_calibrated':False,'physical_anomaly_acceptance':False,
             'operating_envelope_independence_established':False,'deployment_authorized':False,
             'limits':'Fresh source seeds and whole-clean controls improve the benchmark; simulated sensor corruption remains distinct from process health. No physical performance claim or generalization certification.'}
    for split in ('validation','test'):
        rows=card['evaluation'][split]['episodes']
        whole=[r for r in rows if not next(e for e in episodes if e['episode_id']==r['episode_id'])['labels'].count(True)]
        alarms=sum(r['confusion']['FP']>0 for r in whole)
        summary['whole_normal'][split]={'episodes':len(whole),'episodes_with_alarm':alarms,
            'episode_false_alarm_rate':alarms/len(whole) if whole else None}
    (evaluation/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    collect(a.plan,a.output);result=evaluate(a.output)
    print(json.dumps({'total_steps':result['total_steps'],'whole_normal':result['whole_normal']}))
