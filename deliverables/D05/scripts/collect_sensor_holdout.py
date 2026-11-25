"""D-only NEW sensor-model holdout through unchanged B collector; not D07 blind."""
import argparse,copy,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--original-dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-offset',type=int,default=40000);p.add_argument('--exclude-dataset',type=Path,action='append',default=[]);a=p.parse_args()
    source=a.source.resolve();original=json.loads((a.original_dataset/'manifest.json').read_text());model_freeze=json.loads((a.models/'model_freeze.json').read_text())
    for f in model_freeze['models'].values():
        if sha(a.models/f['path'])!=f['sha256']:raise ValueError('frozen model changed')
    a.output.mkdir(parents=True,exist_ok=False);config=copy.deepcopy(original['config']);specs=[]
    for row in config['episodes']:
        if row['split']!='test':continue
        r=copy.deepcopy(row);r['seed']+=a.seed_offset;r['episode_id']='d25_new_sensor_'+str(r['seed'])
        if r['environment_id']=='MAbEnvGym':r['initial_buffer_offset_dm']=float(np.random.default_rng(r['seed']).uniform(-.3,.3))
        specs.append(r)
    plan={'schema_version':'DNewSensorHoldout/1.0','specs':specs,'original_config_sha256':original['frozen_config_sha256'],
        'model_freeze_sha256':sha(a.models/'model_freeze.json'),'collector_sha256':sha(source/'data_generation/collect.py'),
        'seed_offset':a.seed_offset,'new_mab_offset_range_dm':[-.3,.3],'formal_D07_blind':False,'model_training_allowed':False,'labels_used_for_tuning':False}
    write(a.output/'collection_plan.json',plan)
    # This imports and RUNS B code read-only; writes go to D's output only.
    sys.path.insert(0,str(source));from data_generation.collect import collect_episode
    manifest={'schema_version':'DNewSensorHoldout/1.0','config':config,'episodes':[],'complete':False,'plan_sha256':sha(a.output/'collection_plan.json')}
    existing={e['initial_internal_state_sha256'] for e in original['episodes']}
    for extra in a.exclude_dataset:
        existing.update(e['initial_internal_state_sha256'] for e in json.loads((extra/'manifest.json').read_text())['episodes'])
    try:
        for spec in specs:
            report=collect_episode(a.output,config,spec)
            if report['initial_internal_state_sha256'] in existing:raise ValueError('source initial state reused')
            existing.add(report['initial_internal_state_sha256']);manifest['episodes'].append(report)
            write(a.output/'manifest.json',manifest);print(report['episode_id'],report['recorded_steps'],flush=True)
        manifest['complete']=True
    except Exception:manifest['failure_traceback']=traceback.format_exc();raise
    finally:write(a.output/'manifest.json',manifest)
if __name__=='__main__':main()
