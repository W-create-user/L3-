"""Review B's fixed R03 supplement; optional offline sensor-bias IF benchmark.

Never enables a physical rule. Input files and model weights stay local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from l3_system.monitoring.anomaly_detector import AnomalyDetector

SOURCE_COMMIT='6e2b62bbe61f2bee690d55b045ee12d6a6e4497a'
EXPECTED_MANIFEST_SHA256='394c5b992ff0b9f65192953abefed65b2ce2032ece5e4f727afc8d9a5da2a2c0'
SOURCE_INDICES=list(range(1,39))  # Exclude observation[0], a step/time coordinate.


def read(path):
    def invalid(value):raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def safe_path(root,name):
    require(isinstance(name,str) and bool(name),'relative path required')
    path=(root/name).resolve()
    require(path.is_relative_to(root),'path escapes evidence root')
    return path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric39(value):
    array=np.asarray(value)
    require(array.shape==(39,) and array.dtype.kind in 'fiu' and np.isfinite(array).all(),
            'finite numeric 39-variable vector required')
    return array.astype(float)


def candidate_status(observation):
    values=numeric39(observation);reference,current=values[5:7]
    if reference<=0 or 1.10*reference>=20:return 'unsupported_reference_or_censoring'
    return 'candidate_breach' if current>1.10*reference else 'within_candidate_band'


def review(package_root,evidence_root):
    package=Path(package_root).resolve();evidence=Path(evidence_root).resolve()
    require(digest(package/'manifest.json')==EXPECTED_MANIFEST_SHA256,'package differs from fixed reviewed commit')
    manifest=read(package/'manifest.json')
    require(manifest['algorithm']=='SHA256','SHA256 manifest required')
    for name,want in manifest['files'].items():
        require(digest(safe_path(package,name))==want,f'manifest hash mismatch: {name}')
    rule=read(package/'rule_candidate.json')
    require(rule['runtime_enabled'] is False and rule['disposition_proposal']['approved'] is False,
            'unapproved physical candidate must remain disabled')
    require(rule['backend']=='AtropineEnvGym' and rule['layout']=='atropine-default39-v1'
            and rule['observation_dimension']==39 and rule['candidate_relative_limit']==.1,
            'unexpected candidate scope or threshold; new review required')
    require(rule['variables']['current']['index']==6 and rule['variables']['reference']['index']==5,
            'candidate coordinates changed')
    for variable in rule['variables'].values():
        require(variable['normalized'] is False and variable['steady_state_subtracted'] is False,
                'E coordinates must be absolute and unnormalized')
    require(rule['applicable_config']['normalize'] is False,'normalized candidate unsupported')
    for source in rule['source_evidence'].values():
        require(digest(safe_path(evidence,source['path']))==source['file_sha256'],'fixed source evidence mismatch')
    cases=read(package/'validation_cases.json')
    require(len(cases)==2,'expected two constructed cases')
    for case in cases:
        require(case['origin']=='manually_constructed_state_not_real_trajectory','case origin changed')
        require(candidate_status(case['observation'])==case['expected_candidate_status'],'candidate case mismatch')
        require(case['actual']['runtime_enabled'] is False and case['actual']['operational_decision'] is None,
                'candidate used as operational authorization')
    data=read(package/'sensor_benchmark/episodes.json')
    require(data['physical_process_labels_available'] is False and data['physical_process_iforest_ready'] is False,
            'physical health labels remain unavailable')
    require(data['horizon_steps']==30 and data['sampling_period_min']==10,'unexpected acquisition configuration')
    require(digest(evidence/'experiments/collect_r03_sensor_evidence.py')==data['collector_sha256'],'collector hash mismatch')
    episodes=[];ids=set();seeds=set();initials=set();raw_hashes=set();groups={};positives=0
    expected={'train':set(range(1301,1307)),'validation':set(range(1401,1404)),'test':set(range(1501,1504))}
    actual={s:set() for s in expected}
    for item in data['episodes']:
        eid,group,split,seed=(item[k] for k in ('episode_id','group_id','split','seed'))
        require(isinstance(eid,str) and eid not in ids and isinstance(group,str) and bool(group),'unique IDs/groups required')
        require(split in expected and type(seed)is int and seed in expected[split] and seed not in seeds,'unexpected split/seed')
        require(group not in groups or groups[group]==split,'lineage crosses splits')
        require(item['initial_internal_state_sha256'] not in initials,'duplicate initial state')
        require(item['complete'] is True and item['reset_count']==1 and item['recorded_steps']==30
                and item['requested_steps']==30 and item['process_health_label'] is None,'incomplete or physically relabelled episode')
        if split=='train':require(item['fault'] is None,'training must be sensor-uncorrupted')
        else:
            fault=item['fault']
            require(fault['target']=='delivered_observation[6]' and fault['type']=='multiplicative_sensor_bias'
                    and fault['multiplier']==1.2 and fault['first_step']==10 and fault['last_step']==20,
                    'unexpected injection schedule')
        ids.add(eid);seeds.add(seed);initials.add(item['initial_internal_state_sha256']);groups[group]=split;actual[split].add(seed)
        path=safe_path(package/'sensor_benchmark',item['raw_trajectory'])
        require(digest(path)==item['trajectory_sha256'],'trajectory hash mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows)==31,'expected one reset plus thirty steps')
        observations=[];labels=[];raws=[]
        for step,row in enumerate(rows):
            require(row['episode_id']==eid and type(row['step'])is int and row['step']==step
                    and row['sim_time_min']==step*10,'discontinuous or mixed episode')
            require(row['process_health_label'] is None,'physical labels must remain null')
            raw=numeric39(row['environment_observation']);delivered=numeric39(row['delivered_observation'])
            active=split!='train' and 10<=step<=20
            require(type(row['sensor_fault_label'])is int and row['sensor_fault_label']==int(active),'label/schedule mismatch')
            expected_observation=raw.copy()
            if active:expected_observation[6]*=1.2
            require(np.array_equal(delivered,expected_observation),'undeclared sensor alteration')
            raws.append(raw.tolist())
            if step==0:
                require(row.get('kind')=='reset','initial reset required');continue
            require(row['done'] is False and row['terminated'] is False and row['truncated'] is False
                    and row['source_step_count']==step,'early terminal or advancement mismatch')
            require(row['requested_action']==row['actual_action']==[0.,0.,0.,0.],'unexpected action')
            require(row['label_source']=='declared_injection_schedule_not_threshold_or_reward','unexpected label origin')
            observations.append(delivered[SOURCE_INDICES].tolist());labels.append(bool(active));positives+=int(active)
        signature=hashlib.sha256(json.dumps(raws).encode()).hexdigest()
        require(signature not in raw_hashes,'duplicate full environment trajectory');raw_hashes.add(signature)
        episodes.append(dict(episode_id=eid,lineage_group=group,split=split,observations=observations,labels=labels,
            label_scope='sensor_fault',label_source='B declared post-step sensor injection schedule; not physical health'))
    require(actual==expected and len(episodes)==12 and positives==66,'dataset membership/count mismatch')
    receipt={'source_commit':SOURCE_COMMIT,'manifest_sha256':digest(package/'manifest.json'),
        'episodes_sha256':digest(package/'sensor_benchmark/episodes.json'),'manifest_files_verified':len(manifest['files']),
        'episode_counts':{s:len(actual[s]) for s in expected},'successful_steps':360,'sensor_fault_positive_steps':positives,
        'source_observation_indices':SOURCE_INDICES,'excluded_feature':'observation[0] clock; reset rows; all metadata/raw audit channels',
        'label_scope':'sensor_fault','normal_meaning':'no declared artificial sensor corruption, not confirmed physical health',
        'runtime_enabled':False,'deployment_authorized':False,'physical_process_labels_available':False,
        'candidate_cases_checked':len(cases),'independence_limit':'Distinct declared seeds/initial hashes/trajectories; statistical independence and broad operating coverage not established'}
    return episodes,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True);parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--train-sensor-benchmark',action='store_true')
    args=parser.parse_args()
    require(not args.output.exists(),'output must not already exist')
    episodes,receipt=review(args.package_root,args.evidence_root)
    args.output.mkdir(parents=True)
    if args.train_sensor_benchmark:
        # Fixed before inspecting test results; no hyperparameter search on this test set.
        detector=AnomalyDetector(window_size=4,seed=1201)
        card=detector.fit(episodes,label_scope='sensor_fault',receipt=receipt)
        detector.save(args.output/'model')
        receipt['evaluation']=card['evaluation'];receipt['threshold']=card['threshold']
        receipt['threshold_source']='validation_only';receipt['model_dimension']=card['dimension']
        receipt['feature_window_records']=4
    (args.output/'review.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'review_passed':True,'trained_sensor_benchmark':args.train_sensor_benchmark,
                      'physical_rule_enabled':False,'output':str(args.output)}))

if __name__=='__main__':main()
