"""Continuous 100-step episodes for D; never stitch across resets or invent labels."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from integration_abd_1103 import EnvironmentAdapter, ExecutionAgent
from integration_abd_1103.demo_actual_a import load_monitor
from data_collection_1104.provenance import runtime_manifest, sha256
from data_collection_1104.recorder import json_safe, timestamp, write_json
from data_collection_1104.run_collection import state_snapshot, internal_initial_state


def action_for_step(adapter, scenario, step):
    if scenario == 'legal_input_change' and 31 <= step <= 60:
        return (0.25 * adapter.action_space.high).astype(adapter.action_space.dtype), 'positive_legal_offset'
    if scenario == 'legal_input_change' and 61 <= step <= 90:
        return (0.25 * adapter.action_space.low).astype(adapter.action_space.dtype), 'negative_legal_offset'
    return np.zeros(adapter.action_space.shape, dtype=adapter.action_space.dtype), 'USS_zero_offset'


def collect_episode(output, name, seed, steps, adapter, monitor, dt):
    folder = output / 'episodes' / name
    folder.mkdir(parents=True)
    executor = ExecutionAgent(adapter, folder / 'B_events.jsonl')
    monitor.clear_logs()
    executor.reset(seed=seed)
    write_json(folder / 'initial_conditions.json', internal_initial_state(adapter, state_snapshot(adapter), seed))
    dim = int(np.prod(adapter.observation_space.shape))
    actions = int(np.prod(adapter.action_space.shape))
    from smpl.envs.helpers.constants import USS
    transition_fields = [
        'episode_id','scenario','seed','transition_index','task_id','request_id',
        'step_count_before','step_count','sim_time_min','input_condition','input_changed',
        'approved','executed','success','reward','done','terminated','truncated',
        'termination_flag_source','environment_failure_observed','state_anomaly_ground_truth',
        'preclip_source_valid','clipped_field_count','collector_horizon_reached','wall_duration_ms',
    ] + [f'action_{i:02d}' for i in range(actions)] + [f'actual_input_mL_min_{i:02d}' for i in range(actions)] + [f'obs_{i:03d}' for i in range(dim)]
    state_fields = ['episode_id','seed','step_count','sim_time_min','source'] + [f'obs_{i:03d}' for i in range(dim)]
    rows = []
    stop_reason = 'requested_horizon'
    previous_condition = None
    with (folder / 'transitions.csv').open('w',encoding='utf-8-sig',newline='') as tf, \
            (folder / 'observations.csv').open('w',encoding='utf-8-sig',newline='') as sf, \
            (folder / 'raw_events.jsonl').open('w',encoding='utf-8') as jf, \
            (folder / 'preclip_observations.csv').open('w',encoding='utf-8-sig',newline='') as pf:
        tw = csv.DictWriter(tf,fieldnames=transition_fields); tw.writeheader()
        sw = csv.DictWriter(sf,fieldnames=state_fields); sw.writeheader()
        pw = csv.DictWriter(pf,fieldnames=state_fields); pw.writeheader()
        initial = executor.get_observation()
        sw.writerow({'episode_id':name,'seed':seed,'step_count':0,'sim_time_min':0,'source':'reset',
                     **{f'obs_{i:03d}':v for i,v in enumerate(initial)}})
        for index in range(1,steps+1):
            before = state_snapshot(adapter)
            task = {'task_id':f'B-D-1105-{name}-{index:04d}', 'task_type':'control_cycle',
                    'description':'R-03 development data collection; physical safety not labeled'}
            action, condition = action_for_step(adapter, name, index)
            started = time.perf_counter()
            try:
                request = executor.prepare_action(task, action=action)
                review = monitor.check_request(request, request['observation'])
                if review['decision'] == 'pass':
                    result = executor.execute_action(request, review)
                else:
                    result = executor.reject_action_request(request, review)
                duration = (time.perf_counter()-started)*1000
                after = state_snapshot(adapter)
                if not result['executed']:
                    jf.write(json.dumps(json_safe({'event':'candidate_rejected','request':request,
                        'review':review,'result':result,'before':before,'after':after}),ensure_ascii=False,allow_nan=False)+'\n')
                    stop_reason = 'unexpected_monitor_rejection'
                    break
                if 'terminated' not in result or 'truncated' not in result:
                    raise RuntimeError('Adapter must expose explicit derived termination flags')
                preclip_valid = adapter.step_count == before['step_count']+1
                preclip = np.asarray(adapter._env.previous_observation).reshape(-1).tolist() if preclip_valid else None
                observed = result['observation']
                clipped = [i for i,(raw,public) in enumerate(zip(preclip or [],observed or []))
                           if raw != public]
                event = {'event':'transition','timestamp':timestamp(),'transition_index':index,
                         'request':request,'review':review,'result':result,'before':before,'after':after,
                         'preclip_observation':preclip,'preclip_source_valid':preclip_valid,
                         'clipped_indices':clipped,'input_condition':condition,'wall_duration_ms':duration}
                jf.write(json.dumps(json_safe(event),ensure_ascii=False,allow_nan=False)+'\n')
                row = {'episode_id':name,'scenario':name,'seed':seed,'transition_index':index,
                       'task_id':result['task_id'],'request_id':result['request_id'],
                       'step_count_before':before['step_count'],'step_count':after['step_count'],
                       'sim_time_min':after['step_count']*dt,'input_condition':condition,
                       'input_changed':condition!=previous_condition and previous_condition is not None,
                       'approved':result['safety_decision']['approved'],'executed':True,
                       'success':result['success'],'reward':result['reward'],'done':result['done'],
                       'terminated':result['terminated'],'truncated':result['truncated'],
                       'termination_flag_source':result['termination_flag_source'],
                       'environment_failure_observed':not result['success'],
                       'state_anomaly_ground_truth':'',
                       'preclip_source_valid':preclip_valid,'clipped_field_count':len(clipped),
                       'collector_horizon_reached':index==steps and not result['done'],
                       'wall_duration_ms':duration}
                row.update({f'action_{i:02d}':v for i,v in enumerate(request['action'])})
                row.update({f'actual_input_mL_min_{i:02d}':float(USS[i]+request['action'][i]) for i in range(actions)})
                row.update({f'obs_{i:03d}':v for i,v in enumerate(observed or [])})
                tw.writerow(row)
                if observed is not None:
                    sw.writerow({'episode_id':name,'seed':seed,'step_count':after['step_count'],
                        'sim_time_min':after['step_count']*dt,'source':'step' if result['success'] else 'failed_step',
                        **{f'obs_{i:03d}':v for i,v in enumerate(observed)}})
                if preclip is not None:
                    pw.writerow({'episode_id':name,'seed':seed,'step_count':after['step_count'],
                        'sim_time_min':after['step_count']*dt,'source':'upstream_previous_observation_before_clipping',
                        **{f'obs_{i:03d}':v for i,v in enumerate(preclip)}})
                for stream in [tf,sf,jf,pf]: stream.flush()
                rows.append(row)
                previous_condition = condition
                if index == 1 or index % 20 == 0 or result['done']:
                    print(f"{name}: step={after['step_count']} reward={result['reward']} done={result['done']}",flush=True)
                if not result['success'] or result['done']:
                    stop_reason = 'environment_failure' if not result['success'] else 'environment_done'
                    break
            except Exception as error:
                jf.write(json.dumps(json_safe({'event':'exception','timestamp':timestamp(),
                    'transition_index':index,'before':before,'after':state_snapshot(adapter),
                    'error':str(error),'traceback':traceback.format_exc()}),ensure_ascii=False,allow_nan=False)+'\n')
                jf.flush()
                stop_reason = 'exception'
                break
    monitor.export_logs(folder / 'D_reviews.jsonl')
    summary = {'episode_id':name,'seed':seed,'requested_steps':steps,'recorded_steps':len(rows),
               'reset_count':1,'stop_reason':stop_reason,
               'full_continuous_horizon':len(rows)==steps and all(row['success'] for row in rows),
               'terminated_rows':sum(row['terminated'] for row in rows),
               'truncated_rows':sum(row['truncated'] for row in rows),
               'state_anomaly_ground_truth_available':False,
               'input_design':'no injected changes' if name.startswith('baseline') else 'legal input changes, not a labeled dangerous condition'}
    write_json(folder / 'summary.json', summary)
    return summary, rows


def validate_episode(folder, expected_rows):
    with (folder / 'transitions.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    with (folder / 'observations.csv').open(encoding='utf-8-sig',newline='') as stream:
        observations=list(csv.DictReader(stream))
    events=[json.loads(line) for line in (folder/'raw_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    transitions=[e for e in events if e['event']=='transition']
    assert len(rows)==len(transitions)==expected_rows
    previous=json.loads((folder/'initial_conditions.json').read_text(encoding='utf-8'))['state']['observation']
    for index,(row,event) in enumerate(zip(rows,transitions),1):
        result=event['result']
        assert event['before']['observation']==previous
        assert int(row['transition_index'])==index
        assert row['request_id']==result['request_id']
        assert int(row['step_count'])==result['step_count']
        assert row['terminated'].lower()==str(result['terminated']).lower()
        assert row['truncated'].lower()==str(result['truncated']).lower()
        assert bool(result['done'])==bool(result['terminated'] or result['truncated'])
        assert float(row['reward'])==result['reward']
        for i,v in enumerate(result['action']): assert float(row[f'action_{i:02d}'])==v
        for i,v in enumerate(result['observation']): assert float(row[f'obs_{i:03d}'])==v
        previous=result['observation']
    assert len(observations)==1+len(rows)
    return {'episode_id':folder.name,'status':'PASS','rows':len(rows),'observation_rows':len(observations)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--monitor-file')
    args=parser.parse_args()
    if not 1<=args.steps<=100: parser.error('This verified layout supports 1..100 requested steps')
    project=Path(__file__).resolve().parents[1]
    monitor_file=Path(args.monitor_file or project/'external_D/pr1_8c105883/source/agents/monitoring_agent.py').resolve()
    output=project/'data_1105/runs'/(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True)
    with EnvironmentAdapter(max_steps=-1) as adapter:
        monitor,source=load_monitor(adapter.action_space,monitor_file)
        assert source=='D_provided_source' and not monitor.demo_mode
        from smpl.envs.models.config import dt
        from smpl.envs.helpers.constants import USS
        a_root=project/'integration_abd_1103/member_A_11_02'
        metadata=runtime_manifest(adapter.source_dir,a_root,monitor_file,project)
        metadata.update(created_at=timestamp(),requested_steps=args.steps,
            max_steps=-1,max_steps_source='official public constructor option; only time limit disabled',
            normalize=False,uss_subtracted=True,USS=USS.tolist(),sampling_interval_minutes=dt,
            observation_shape=list(adapter.observation_space.shape),action_shape=list(adapter.action_space.shape),
            action_low=adapter.action_space.low.tolist(),action_high=adapter.action_space.high.tolist(),
            task_source='B data acquisition tasks; A Task IDs for prior collaboration not invented here',
            D_source=source,raw_termination_api='legacy done plus info.timeout; no native Gymnasium flags',
            collector_code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        write_json(output/'manifest.json',metadata)
        assert adapter.action_space.contains(np.zeros(adapter.action_space.shape,dtype=adapter.action_space.dtype))
        summaries=[]; baseline_rows=[]; validations=[]
        for name,seed in [('baseline_seed1101',1101),('baseline_seed1102',1102),('legal_input_change',1101)]:
            summary,rows=collect_episode(output,name,seed,args.steps,adapter,monitor,dt)
            summaries.append(summary)
            validations.append(validate_episode(output/'episodes'/name,len(rows)))
            if name.startswith('baseline'):
                baseline_rows.extend(row for row in rows if row['success'] and not row['done'])
        if baseline_rows:
            field_schema=json.loads((project/'integration_abd_1103/exports_D/20261104_123730_562116/observation_schema.json').read_text(encoding='utf-8'))
            clean_fields=[]
            for field in field_schema['fields']:
                item={k:v for k,v in field.items() if k not in ('reset_example','step_example')}
                clean_fields.append(item)
            write_json(output/'observation_schema.json',{'fields':clean_fields,'source':'verified unchanged observation layout; see model hashes','safety_thresholds':False})
            with (output/'baseline_empirical_ranges.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['index','name','unit','count','min','max','mean','std','p01','p99','interpretation'])
                writer.writeheader()
                for field in clean_fields:
                    values=np.asarray([row[f"obs_{field['index']:03d}"] for row in baseline_rows],dtype=float)
                    writer.writerow({'index':field['index'],'name':field['name'],'unit':field['unit'],'count':len(values),
                        'min':values.min(),'max':values.max(),'mean':values.mean(),'std':values.std(),
                        'p01':np.quantile(values,0.01),'p99':np.quantile(values,0.99),
                        'interpretation':'empirical coverage under two seeded zero-offset runs; NOT an industrial safety threshold'})
    write_json(output/'episode_summaries.json',summaries)
    with (output/'episode_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    verification={'status':'PASS','data_validation':validations,'episodes':summaries,
                  'baseline_100_step_episodes':sum(s['full_continuous_horizon'] and s['episode_id'].startswith('baseline') for s in summaries),
                  'range_scope':'empirical model-run ranges only','industrial_safety_thresholds_available':False}
    write_json(output/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False),flush=True)
    print('Evidence:',output,flush=True)


if __name__=='__main__': main()
