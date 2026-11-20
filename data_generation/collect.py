"""Collect B05 whole-episode datasets; keep labels and sources out of model features."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from l3_system.execution import create_adapter, ExecutionAgent
from l3_system.execution.controllers import BufferLevelPID, SpaceMidpointPolicy
from l3_system.agents.monitoring_agent import MonitoringAgent
from .transforms import sensor_view, feature_payload

ROOT=Path(__file__).resolve().parents[1]

def json_write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def render_visual(path,history,profile,window):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    values=history[-window:]
    figure,axis=plt.subplots(figsize=(4,2),dpi=80)
    axis.plot([row[0] for row in values],[np.nan if row[1] is None else row[1]*profile['physical_scale'] for row in values],color='#2463a3')
    axis.set_xlabel('simulation time (min)');axis.set_ylabel(profile['unit'])
    figure.tight_layout();figure.savefig(path,metadata={'Software':'B05 causal synthetic visual'});plt.close(figure)

def collect_episode(root,config,spec):
    eid=spec['episode_id'];folder=root/eid;folder.mkdir();visuals=folder/'visuals';visuals.mkdir()
    profile=config['profiles'][spec['environment_id']]
    report={'episode_id':eid,'environment_id':spec['environment_id'],'seed':spec['seed'],'split':spec['split'],
            'reset_count':1,'horizon_steps':spec['horizon_steps'],'complete':False,'physical_health_label':None,
            'fault':spec['fault'],'samples_file':eid+'/samples.jsonl','steps_file':eid+'/steps.jsonl',
            'initial_state_intervention':None,'model_observation_is_not_field_sensor':True}
    rng=np.random.default_rng(spec['seed']^0xB052020)
    policy_config=json.loads((ROOT/'configs/controllers/mab_buffer_pi.json').read_text())
    with create_adapter(spec['environment_id'],max_steps=max(60,spec['horizon_steps']+1)) as adapter:
        if Path(adapter.source_dir).resolve()!=(ROOT/'vendor/smpl').resolve():raise ValueError('Frozen protocol requires the recorded vendored model source, not SMPL_SOURCE_DIR override')
        if adapter.metadata['sampling_period']!={'value':profile['sample_period_min'],'unit':'min'}:raise ValueError('Sampling period source mismatch')
        if adapter.observation_space.shape!=(profile['observation_size'],):raise ValueError('Observation profile mismatch')
        dictionary=list(csv.DictReader((ROOT/'docs/data_dictionary.csv').open()))
        field=next(row for row in dictionary if row['backend']==spec['environment_id'] and row['kind']=='observation' and int(row['index'])==profile['sensor_index'])
        if field['physical_unit']!=profile['unit'] or float(field['scale'])!=profile['physical_scale']:raise ValueError('Sensor unit/scale does not match source dictionary')
        executor=ExecutionAgent(adapter,log_path=folder/'B_events.jsonl')
        if spec['environment_id']=='MAbEnvGym':
            state=adapter._env.steady_observations.copy();state[17]+=spec['initial_buffer_offset_dm']/adapter._xscale[17]
            adapter.reset(seed=spec['seed'],initial_state=state)
            report['initial_state_intervention']={'kind':'actual_simulator_initial_state_offset','index':17,
                'physical_unit':'dm','amplitude':spec['initial_buffer_offset_dm'],'sim_time_min':0,
                'basis':'mab_helpers buffer h=x[17], xscale[17]=3; bounded by prior B04 tested offsets',
                'not_physical_health_ground_truth':True}
            internal=np.asarray(adapter._env.Xi[0],dtype=np.float64)
            policy=BufferLevelPID(adapter,policy_config)
        else:
            executor.reset(seed=spec['seed']);internal=np.asarray(adapter._env.xk.full(),dtype=np.float64)
            policy=SpaceMidpointPolicy()
        policy.reset();initial=executor.get_observation()
        fingerprint=hashlib.sha256(internal.tobytes()).hexdigest()
        report.update(initial_internal_state_sha256=fingerprint,group_id=spec['environment_id']+':'+fingerprint,
                      adapter_metadata=adapter.metadata,initial_observation=initial,
                      feature_profile=profile,stopped_by='declared_horizon_not_natural_terminal')
        monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
        semantic=[];history=[];positives=0
        with (folder/'steps.jsonl').open('w') as sf,(folder/'samples.jsonl').open('w') as mf,(folder/'runtime_stdout.log').open('w') as runtime_log:
            for step in range(1,spec['horizon_steps']+1):
                before=executor.get_observation();version=adapter.state_version
                action=policy.predict(before,adapter.action_space)
                task={'task_id':f'{eid}_{step}','task_type':'control_cycle'}
                with contextlib.redirect_stdout(runtime_log):result=executor.execute_assignment(task,monitor.check_request,action=action)
                runtime_log.flush()
                policy.commit(bool(result['success']))
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                seconds=step*profile['sample_period_min']*60
                record={'schema_version':'D01Step/0.1','episode_id':eid,'step_index':step,'simulation_time':seconds,
                        'simulation_time_unit':'s','collected_at':timestamp,'observation_before':before,
                        'observation_after':result['observation'],'candidate_action':np.asarray(action).tolist(),
                        'executed_action':result['action'] if result['executed'] else None,'executed':result['executed'],
                        'state_version_before':version,'state_version_after':adapter.state_version,
                        'reward':result['reward'],'terminated':bool(result.get('terminated',False)),
                        'truncated':bool(result.get('truncated',False)),'execution_status':result['execution_status'],
                        'failure_reason':result['failure_reason']}
                sf.write(json.dumps(record,allow_nan=False)+'\n');sf.flush()
                if not result['success'] or not result['executed']:raise RuntimeError(f'{eid} failed at step {step}; partial evidence preserved')
                delivered,mask,label,event=sensor_view(result['observation'],index=profile['sensor_index'],step=step,fault=spec['fault'],rng=rng)
                history.append((seconds/60,delivered[profile['sensor_index']]))
                image=None;image_hash=None
                if step%config['visual_stride_steps']==0:
                    local='visuals/'+f'{step:04d}.png';path=folder/local
                    render_visual(path,history,profile,config['visual_window_steps']);image=eid+'/'+local
                    image_hash=hashlib.sha256(path.read_bytes()).hexdigest()
                features=feature_payload(delivered,mask,indices=profile['feature_indices'],scale=profile['physical_scale'],unit=profile['unit'])
                sample={'sample_id':f'{eid}:{step}','episode_id':eid,'step':step,'sim_time_min':seconds/60,
                        'environment_id':spec['environment_id'],'features':features,'delivered_observation':delivered,
                        'missing_observation_mask':mask,'labels':label,'event':event,
                        'visual':{'path':image,'sha256':image_hash,'synthetic_visual':True,'derived_from':'causal_delivered_series',
                                  'window_end_step':step,'independent_information':False},
                        'modality_missing':{'numeric':all(features['missing_mask']),'text':False,'visual':image is None},
                        'raw_source':{'path':eid+'/steps.jsonl','line':step,'coordinate':adapter.metadata['observation_coordinate']},
                        'collector_horizon_reached':step==spec['horizon_steps']}
                mf.write(json.dumps(sample,allow_nan=False)+'\n');mf.flush()
                semantic.append({'step':step,'before':before,'after':result['observation'],'action':record['executed_action'],
                                 'delivered':delivered,'features':features,'label':label,'image_sha256':image_hash})
                positives+=label['sensor_fault_label'];report['recorded_steps']=step
                if result['done'] and step<spec['horizon_steps']:raise RuntimeError('Early terminal; full episode requirement not met')
        report.update(complete=True,semantic_sha256=digest(semantic),sensor_positive_steps=positives,
                      state_version_after=adapter.state_version,step_calls=adapter.step_calls,
                      source_trace_sha256=hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest())
    json_write(folder/'episode.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());from .validation import validate_config
    validate_config(config);a.output.mkdir(parents=True,exist_ok=False)
    ids=[e['episode_id'] for e in config['episodes']]
    if len(ids)!=len(set(ids)) or any('/' in e or '..' in e for e in ids):raise ValueError('Canonical unique episode IDs required')
    manifest={'schema_version':'B05Dataset/1.0','dataset_id':config['dataset_id'],'frozen_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
              'config':config,'source_kind':'real_research_simulation_plus_synthetic_modalities',
              'physical_health_labels_available':False,'episodes':[],'complete':False,
              'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependency_versions':{name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','casadi','gym','MPCTools']},
              'source_files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
                  'data_generation/collect.py','data_generation/transforms.py','data_generation/validation.py',
                  'l3_system/execution/environment_adapter.py','l3_system/execution/mab_adapter.py',
                  'l3_system/execution/execution_agent.py','l3_system/execution/controllers/policies.py',
                  'configs/controllers/mab_buffer_pi.json','vendor/smpl/SOURCE_VERSION.json']}}
    # Bind actual runtime/model files and configuration arrays, not only the upstream-version note.
    extra=[ROOT/'docs/data_dictionary.csv',ROOT/'configs/environments/dictionary_provenance.json']
    extra+=list((ROOT/'l3_system/execution').rglob('*.py'))
    extra+=list((ROOT/'vendor/smpl/smpl').rglob('*.py'))
    for config_dir in ['atropineenv','mabenv']:
        extra+=[path for path in (ROOT/'vendor/smpl/smpl/configdata'/config_dir).iterdir() if path.is_file()]
    for path in sorted(extra):manifest['source_files_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    json_write(a.output/'manifest.json',manifest)
    (a.output/'frozen_config.json').write_bytes(a.config.read_bytes())
    try:
        for spec in config['episodes']:
            manifest['active_episode_id']=spec['episode_id'];json_write(a.output/'manifest.json',manifest)
            report=collect_episode(a.output,config,spec);manifest['episodes'].append(report)
            json_write(a.output/'manifest.json',manifest);print(report['episode_id'],'complete',flush=True)
        groups={}
        for report in manifest['episodes']:
            group=report['group_id']
            if group in groups and groups[group]!=report['split']:raise RuntimeError('Initial-state lineage leaked across splits')
            groups[group]=report['split']
        manifest['complete']=True;manifest['active_episode_id']=None
        manifest['semantic_sha256']=digest({r['episode_id']:r['semantic_sha256'] for r in manifest['episodes']})
        manifest['freeze_sha256']=digest({'config':manifest['frozen_config_sha256'],'source':manifest['source_files_sha256'],'samples':manifest['semantic_sha256']})
    except Exception:
        manifest['failure_traceback']=traceback.format_exc();raise
    finally:json_write(a.output/'manifest.json',manifest)

if __name__=='__main__':main()
