"""Recompute D24 mAb outcomes from raw steps; never certify plant performance."""
import hashlib,json,math
from pathlib import Path
import numpy as np

ARMS=('traditional_controller','single_agent','four_agent')

def audit(root,source_root=None):
    root=Path(root);plan=json.loads((root/'frozen_plan.json').read_text())
    report=json.loads((root/'summary.json').read_text())
    if plan.get('schema_version')!='D07MAbPaired/1.0' or report.get('schema_version')!='D24MAbSystemComparison/1.0':
        raise ValueError('known frozen protocol/report schemas required')
    if report['pilot'] or len(plan['runs'])!=90:raise ValueError('full 90-run frozen comparison required')
    if hashlib.sha256((root/'frozen_plan.json').read_bytes()).hexdigest()!=report['plan_sha256']:
        raise ValueError('frozen plan changed')
    expected={r['run_id']:r for r in plan['runs']};rows=report['runs']
    if len(expected)!=90 or len(rows)!=90 or set(expected)!={r['run_id'] for r in rows}:
        raise ValueError('missing or duplicate planned runs')
    if source_root is not None:
        for name,want in plan['source_sha256'].items():
            if hashlib.sha256((Path(source_root)/name).read_bytes()).hexdigest()!=want:
                raise ValueError('source changed: '+name)
    action_bounds={}
    for r in rows:
        if r['arm']=='four_agent' and r['started']:
            goal=json.loads((root/r['run_id']/'A_goal_result.json').read_text())['goal_spec']
            allowed=goal['allowed_actions']
            action_bounds[r['pair_id']]=np.array([[allowed[f'model_action_{i}']['range'][side]
                for i in range(9)] for side in ('lower','upper')],dtype=float)
    pairs={};reviewed=[];clip_records=0;violations=0;buffer_clip_records=0;clipped_indices={}
    for r in rows:
        spec=expected[r['run_id']]
        if any(r[k]!=spec[k] for k in ('pair_id','condition','seed','arm','initial_buffer_offset_dm')):
            raise ValueError('run conditions differ from plan')
        folder=root/r['run_id'];steps=json.loads((folder/'steps.json').read_text()) if r['started'] else []
        if len(steps)!=r['step_calls']:raise ValueError('raw step denominator changed')
        streak=0;earliest=None;bad=0;action_list=[];previous_version=None
        for i,s in enumerate(steps,1):
            raw=np.asarray(s['raw_preclip_observation'],dtype=float);action=np.asarray(s['action'],dtype=float)
            if (raw.shape!=(1970,) or action.shape!=(9,) or not np.isfinite(raw).all()
                    or not np.isfinite(action).all() or s['step_count']!=i
                    or previous_version is not None and s['state_version']!=previous_version+1):
                raise ValueError('malformed/noncontiguous raw model trace')
            previous_version=s['state_version'];h=float(raw[17]*3)
            bad+=int(not 0<=h<=50 or s['success'] is not True)
            limits=action_bounds.get(r['pair_id'])
            if limits is None or np.any(action<limits[0]) or np.any(action>limits[1]):
                raise ValueError('candidate action outside matched live model bounds')
            indices=s['info'].get('clipped_indices') or []
            clip_records+=int(bool(indices));buffer_clip_records+=int(17 in indices)
            for index in indices:clipped_indices[str(index)]=clipped_indices.get(str(index),0)+1
            streak=streak+1 if 14.95<=h<=15.05 else 0
            if streak>=3 and earliest is None:earliest=i
            action_list.append(action.tolist())
        if r['success'] and (earliest is None or earliest!=r['step_calls'] or bad):
            raise ValueError('reported goal success does not match raw model trajectory')
        if r['success'] and (not r.get('pending_requests_clear') or r.get('C_resources_clear',True) is not True):
            raise ValueError('successful run retains pending resources')
        violations+=bad
        record={**r,'raw_goal_first_step':earliest,'raw_model_violation_count':bad,'actions':action_list}
        reviewed.append(record);pairs.setdefault(r['pair_id'],{})[r['arm']]=record
    if len(pairs)!=30 or any(set(v)!=set(ARMS) for v in pairs.values()):raise ValueError('incomplete matched arm sets')
    unique=set();same_actions=0;complete_times=[]
    for pair in pairs.values():
        hashes={r.get('initial_internal_state_sha256') for r in pair.values()}
        if len(hashes)!=1 or None in hashes:raise ValueError('unmatched initial internal states')
        unique.update(hashes)
        if all(pair[a]['actions']==pair['traditional_controller']['actions'] for a in ARMS):same_actions+=1
        b=pair['traditional_controller'];c=pair['four_agent']
        if b['success'] and c['success']:
            complete_times.append((b['simulated_makespan_min']-c['simulated_makespan_min'])/b['simulated_makespan_min'])
    groups={}
    for condition in ('nominal','sensor_fault','process_disturbance'):
        groups[condition]={}
        for arm in ARMS:
            rs=[r for r in reviewed if r['condition']==condition and r['arm']==arm]
            if len(rs)!=10:raise ValueError('condition/arm planned denominator must be ten')
            groups[condition][arm]={'planned':len(rs),'started':sum(r['started'] for r in rs),
                'successes':sum(r['success'] for r in rs),'failures':sum(r['started'] and not r['success'] for r in rs),
                'success_rate':sum(r['success'] for r in rs)/sum(r['started'] for r in rs) if any(r['started'] for r in rs) else None,
                'mean_simulated_makespan_min_success_only':float(np.mean([r['simulated_makespan_min'] for r in rs if r['success']])) if any(r['success'] for r in rs) else None,
                'mean_software_wall_seconds':float(np.mean([r['wall_seconds'] for r in rs]))}
    return {'schema_version':'D24MAbRawAudit/1.0','planned_runs':90,'started_runs':sum(r['started'] for r in reviewed),
        'successful_runs':sum(r['success'] for r in reviewed),'raw_step_calls':sum(r['step_calls'] for r in reviewed),
        'paired_groups':30,'unique_initial_state_fingerprints':len(unique),'identical_action_sequence_pairs':same_actions,
        'raw_buffer_model_violations':violations,'records_with_any_model_clipping':clip_records,
        'buffer_level_clipped_records':buffer_clip_records,'clipping_count_by_model_index':clipped_indices,'by_condition':groups,
        'complete_successful_makespan_pairs':len(complete_times),'mean_simulated_makespan_reduction':float(np.mean(complete_times)) if complete_times else None,
        'efficiency_target_20_percent_supported':False,'whole_project_90_percent_claim':False,'energy_metric':None,
        'independent_episode_assumption_verified':False,'physical_safety_certified':False,
        'limits':'Research buffer goal only. Topologies intentionally share the same proportional law; local C has one executor. Simulated goal time is separate from software wall time. No production/plant generalization claim.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.root,a.source_root)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('started_runs','successful_runs','raw_step_calls','paired_groups','mean_simulated_makespan_reduction')}))
