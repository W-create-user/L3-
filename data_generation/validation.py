"""Fail-closed B05 configuration and frozen dataset integrity checks."""
import hashlib
import json
import math
import numpy as np
from pathlib import Path


def validate_config(config):
    if config.get('schema_version')!='B05Collection/1.0':raise ValueError('Unknown B05 schema')
    if config.get('physical_health_label_available') is not False:raise ValueError('No physical health labels are approved')
    ids=set();seeds=set();kinds=set();per_environment={name:set() for name in config['profiles']};splits={'train','validation','test'}
    if type(config.get('visual_stride_steps'))is not int or config['visual_stride_steps']<1:raise ValueError('Invalid visual stride')
    if type(config.get('visual_window_steps'))is not int or config['visual_window_steps']<1:raise ValueError('Invalid causal window')
    for row in config['episodes']:
        eid=row['episode_id'];env=row['environment_id'];seed=row['seed'];split=row['split'];horizon=row['horizon_steps']
        if (not isinstance(eid,str) or not eid or any(c in eid for c in '/\\') or '..' in eid or eid in ids
                or type(seed)is not int or (env,seed) in seeds or split not in splits
                or env not in config['profiles'] or type(horizon)is not int or not 1<=horizon<=60):raise ValueError('Invalid/duplicate episode definition')
        ids.add(eid);seeds.add((env,seed));profile=config['profiles'][env]
        if (type(profile['sensor_index'])is not int or not 0<=profile['sensor_index']<profile['observation_size']
                or any(type(i)is not int or not 0<=i<profile['observation_size'] for i in profile['feature_indices'])):raise ValueError('Invalid variable indices')
        if env=='MAbEnvGym' and (horizon>10 or not -.3<=row.get('initial_buffer_offset_dm',float('inf'))<=.3):raise ValueError('Unverified mAb initial/horizon range')
        fault=row['fault']
        if fault:
            if split=='train':raise ValueError('Training episodes must not contain injected faults')
            kind=fault['kind'];first=fault['first_step'];last=fault['last_step'];kinds.add(kind);per_environment[env].add(kind)
            if kind not in {'bias','noise','missing'} or type(first)is not int or type(last)is not int or not 1<=first<=last<=horizon:raise ValueError('Invalid fault interval')
            key='multiplier' if kind=='bias' else 'sigma'
            if kind!='missing':
                value=fault[key]
                if type(value)not in (int,float) or not math.isfinite(value) or (kind=='noise' and value<=0) or (kind=='bias' and value==1):raise ValueError('Invalid sensor magnitude')
    if kinds!={'bias','noise','missing'} or any(k!={'bias','noise','missing'} for k in per_environment.values()):raise ValueError('Normal plus three predeclared fault types required on each backend')
    for split in splits:
        for env in config['profiles']:
            if not any(r['split']==split and r['environment_id']==env and r['fault'] is None for r in config['episodes']):raise ValueError('Whole-clean controls required for each backend/split')
    return True


def validate_dataset(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());validate_config(manifest['config'])
    if not manifest['complete']:raise ValueError('Partial datasets cannot be frozen')
    if hashlib.sha256((root/'frozen_config.json').read_bytes()).hexdigest()!=manifest['frozen_config_sha256']:raise ValueError('Frozen input changed')
    if json.loads((root/'frozen_config.json').read_text())!=manifest['config']:raise ValueError('Embedded configuration mismatch')
    project=Path(__file__).resolve().parents[1]
    for name,expected_hash in manifest['source_files_sha256'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest()!=expected_hash:raise ValueError('Use the original collector source revision to validate this frozen dataset: '+name)
    expected={r['episode_id']:r for r in manifest['config']['episodes']};reports=manifest['episodes']
    if len(reports)!=len(expected) or set(expected)!={r['episode_id'] for r in reports}:raise ValueError('Missing/duplicate episodes')
    groups={};total=positive=images=0;semantic={}
    from .transforms import feature_payload, sensor_view
    for report in reports:
        spec=expected[report['episode_id']];profile=report['feature_profile'];group=report['group_id']
        if group in groups and groups[group]!=report['split']:raise ValueError('Cross-split group leakage')
        groups[group]=report['split']
        if not report['complete'] or report['reset_count']!=1 or report['recorded_steps']!=spec['horizon_steps']:raise ValueError('Incomplete or stitched trajectory')
        folder=root/report['episode_id'];steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        if len(steps)!=spec['horizon_steps'] or len(samples)!=len(steps):raise ValueError('Trajectory length mismatch')
        if hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest()!=report['source_trace_sha256']:raise ValueError('Source trace hash mismatch')
        causal_history=[];canonical=[];expected_rng=np.random.default_rng(spec['seed']^0xB052020);previous=report['initial_observation']
        for index,(step,sample) in enumerate(zip(steps,samples),1):
            if step['step_index']!=index or sample['step']!=index or sample['episode_id']!=report['episode_id']:raise ValueError('Noncontiguous/misaligned sample')
            if step['simulation_time']!=index*profile['sample_period_min']*60 or sample['sim_time_min']!=step['simulation_time']/60:raise ValueError('Time/unit mismatch')
            if (not step['executed'] or step['execution_status']!='completed' or step['state_version_after']!=step['state_version_before']+1
                    or step['candidate_action']!=step['executed_action']):raise ValueError('Invalid executed transition')
            if step['observation_before']!=previous:raise ValueError('State trajectory discontinuity')
            raw=step['observation_after'];delivered=sample['delivered_observation'];mask=sample['missing_observation_mask'];previous=raw
            wanted_view,wanted_mask,wanted_label,wanted_event=sensor_view(raw,index=profile['sensor_index'],step=index,fault=spec['fault'],rng=expected_rng)
            if (delivered,mask,sample['labels'],sample['event'])!=(wanted_view,wanted_mask,wanted_label,wanted_event):raise ValueError('Injected view or label does not reproduce from declared seed')
            if len(raw)!=profile['observation_size'] or len(delivered)!=len(raw) or len(mask)!=len(raw):raise ValueError('Wrong shape')
            active=bool(spec['fault'] and spec['fault']['first_step']<=index<=spec['fault']['last_step'])
            if sample['labels']['physical_health_label'] is not None or sample['labels']['sensor_fault_label']!=int(active):raise ValueError('Fabricated label')
            sensor=profile['sensor_index']
            for i,value in enumerate(delivered):
                if i!=sensor or not active:
                    if value!=raw[i] or mask[i]:raise ValueError('Undeclared sensor change')
            if active:
                if spec['fault']['kind']=='missing':
                    if delivered[sensor] is not None or not mask[sensor]:raise ValueError('Missing feature was imputed without permission')
                elif delivered[sensor] is None or mask[sensor]:raise ValueError('Unexpected null feature')
                if spec['fault']['kind']=='bias' and delivered[sensor]!=raw[sensor]*spec['fault']['multiplier']:raise ValueError('Bias does not match declaration')
            wanted=feature_payload(delivered,mask,indices=profile['feature_indices'],scale=profile['physical_scale'],unit=profile['unit'])
            if sample['features']!=wanted:raise ValueError('Text/feature leakage or wrong causal projection')
            if sample['modality_missing']['numeric']!=all(wanted['missing_mask']):raise ValueError('Fully missing numeric projection not marked')
            visual=sample['visual'];available=index%manifest['config']['visual_stride_steps']==0
            if sample['modality_missing']['visual'] != (not available):raise ValueError('Visual missing flag wrong')
            if visual['window_end_step']!=index or visual['independent_information'] is not False or visual['synthetic_visual'] is not True:raise ValueError('Visual provenance invalid')
            if available:
                image=root/visual['path']
                if hashlib.sha256(image.read_bytes()).hexdigest()!=visual['sha256']:raise ValueError('Image source/hash mismatch')
                images+=1
            elif visual['path'] is not None:raise ValueError('Unexpected visual')
            canonical.append({'step':index,'before':step['observation_before'],'after':raw,'action':step['executed_action'],
                              'delivered':delivered,'features':sample['features'],'label':sample['labels'],'image_sha256':visual['sha256']})
            total+=1;positive+=int(active)
        from .collect import digest
        if digest(canonical)!=report['semantic_sha256']:raise ValueError('Semantic hash mismatch')
        semantic[report['episode_id']]=report['semantic_sha256']
    from .collect import digest
    if digest(semantic)!=manifest['semantic_sha256']:raise ValueError('Frozen dataset hash mismatch')
    if digest({'config':manifest['frozen_config_sha256'],'source':manifest['source_files_sha256'],'samples':manifest['semantic_sha256']})!=manifest['freeze_sha256']:raise ValueError('Freeze binding mismatch')
    return {'validated':True,'episodes':len(reports),'steps':total,'sensor_positive_steps':positive,'synthetic_visuals':images,
            'physical_health_labels_available':False,'semantic_sha256':manifest['semantic_sha256']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(validate_dataset(a.root),indent=2))
