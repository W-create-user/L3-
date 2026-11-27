"""Recompute B08 development outcomes from pre-clip states and planned pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .run_b08_development import ARMS, ROOT, write


def paired_estimate(baseline,candidate,conditions,*,seed=261126,resamples=2000):
    b=np.asarray(baseline,dtype=float);c=np.asarray(candidate,dtype=float)
    if b.size==0:return None
    if b.shape!=c.shape or len(conditions)!=len(b) or not np.isfinite(b).all() or not np.isfinite(c).all() or np.any(b<=0):
        raise ValueError('Finite matched cycles required')
    groups=[np.flatnonzero(np.asarray(conditions)==kind) for kind in sorted(set(conditions))]
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        indices=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        draws.append(1-float(np.mean(c[indices]))/float(np.mean(b[indices])))
    return {'pairs':len(b),'ratio_of_paired_means_reduction':1-float(np.mean(c))/float(np.mean(b)),
            'stratified_paired_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),
            'resamples':resamples,'bootstrap_seed':seed,'development_only':True,
            'independence_verified':False,'formal_target_acceptance':False}


def audit(root,source_root=ROOT):
    root=Path(root);plan=json.loads((root/'frozen_plan.json').read_text())
    rows=json.loads((root/'run_ledger.json').read_text());planned={r['run_id']:r for r in plan['runs']}
    if plan['schema_version']!='B08Development/1.0' or plan['formal_blind_test'] is not False:
        raise ValueError('Development plan required')
    if len(planned)!=len(plan['runs']) or len(rows)!=len(planned) or len({r['run_id'] for r in rows})!=len(rows) or {r['run_id'] for r in rows}!=set(planned):
        raise ValueError('Missing/duplicate/unplanned receipts')
    for path,sha in plan['source_sha256'].items():
        if hashlib.sha256((Path(source_root)/path).read_bytes()).hexdigest()!=sha:raise ValueError('Changed frozen source: '+path)
    bounds={}
    for row in rows:
        path=root/row['run_id']/'goal_result.json'
        if row['arm']=='observer_four' and path.exists():
            result=json.loads(path.read_text())
            if 'goal_spec' in result:
                allowed=result['goal_spec']['allowed_actions']
                bounds[row['pair_id']]=np.array([[allowed[f'model_action_{i}']['range'][side] for i in range(9)] for side in ('lower','upper')])
    pairs={};steps_total=0;clipping={}
    for row in rows:
        spec=planned[row['run_id']]
        if any(row[k]!=spec[k] for k in ('seed','condition','arm','pair_id','initial_buffer_offset_dm')):
            raise ValueError('Planned condition mismatch')
        folder=root/row['run_id'];path=folder/'steps.jsonl'
        steps=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        if len(steps)!=row['step_calls']:raise ValueError('Step count mismatch')
        streak=0;first=None;version=None;bad=False
        for i,step in enumerate(steps,1):
            raw=np.asarray(step['raw_preclip_observation'],dtype=float)
            action=np.asarray(step['action'],dtype=float)
            if raw.shape!=(1970,) or not np.isfinite(raw).all() or action.shape!=(9,) or not np.isfinite(action).all():raise ValueError('Malformed trace')
            if step['step_count']!=i or (version is not None and step['state_version']!=version+1):raise ValueError('Noncontiguous trace')
            limits=bounds.get(row['pair_id'])
            if limits is None:raise ValueError('No paired live action bounds available for raw audit')
            if np.any(action<limits[0]) or np.any(action>limits[1]):raise ValueError('Executed action outside paired live bounds')
            version=step['state_version'];h=float(raw[17]*3)
            bad=bad or step['success'] is not True
            streak=streak+1 if 14.95<=h<=15.05 and step['success'] is True else 0
            if streak>=3 and first is None:first=i
            for index in step['info'].get('clipped_indices',[]):clipping[str(index)]=clipping.get(str(index),0)+1
        if row['success'] and (bad or first!=len(steps) or row['simulated_makespan_min']!=first*60
                or row['wall_seconds']>90 or not row.get('pending_requests_clear') or not row.get('C_resources_clear',True)):
            raise ValueError('Success does not match complete raw on-time trajectory')
        if row['arm'].startswith('observer') and row['started']:
            events=json.loads((folder/'observer_trace.json').read_text())
            if len(events)!=len(steps):raise ValueError('Observer receipt count differs')
            level=events[0]['estimated_level_dm'] if events else None
            for event,step in zip(events,steps):
                if abs(event['estimated_level_dm']-level)>1e-8:raise ValueError('Observer not driven by acknowledged actions')
                u=np.asarray(step['action'])*_uscale(source_root)
                level+=60*(u[3]-u[7])/(np.pi*2.5**2)
            # No true poststep, label or event schedule is an observer input.
        pairs.setdefault(row['pair_id'],{})[row['arm']]=row
        steps_total+=len(steps)
    for pair in pairs.values():
        if set(pair)!=set(ARMS):raise ValueError('Incomplete arm set')
        if all(r['started'] for r in pair.values()) and len({r['initial_internal_state_sha256'] for r in pair.values()})!=1:
            raise ValueError('Unmatched initial states')
    equivalent_candidate_pairs=0
    for pair in pairs.values():
        left=root/pair['observer_direct']['run_id']/'steps.jsonl'
        right=root/pair['observer_four']['run_id']/'steps.jsonl'
        if left.exists() and right.exists():
            a=[json.loads(s)['action'] for s in left.read_text().splitlines()]
            b=[json.loads(s)['action'] for s in right.read_text().splitlines()]
            equivalent_candidate_pairs+=int(a==b)
    groups={arm:{'planned':sum(s['arm']==arm for s in plan['runs']),
        'started':sum(r['started'] and r['arm']==arm for r in rows),
        'successes':sum(r['success'] and r['arm']==arm for r in rows),
        'not_started':sum(not r['started'] and r['arm']==arm for r in rows),
        'failures':sum(r['started'] and not r['success'] and r['arm']==arm for r in rows),
        'mean_successful_simulated_minutes':float(np.mean([r['simulated_makespan_min'] for r in rows if r['arm']==arm and r['success']])) if any(r['success'] and r['arm']==arm for r in rows) else None}
        for arm in ARMS}
    comparisons={}
    for arm in ARMS[1:]:
        complete=[p for p in pairs.values() if p['baseline_P']['success'] and p[arm]['success']]
        full=list(pairs.values())
        comparisons[arm]={'complete_successful_pairs':paired_estimate(
            [p['baseline_P']['simulated_makespan_min'] for p in complete],
            [p[arm]['simulated_makespan_min'] for p in complete],[p[arm]['condition'] for p in complete]),
            'all_planned_failure_penalty_600_min':paired_estimate(
            [p['baseline_P']['simulated_makespan_min'] if p['baseline_P']['success'] else 600 for p in full],
            [p[arm]['simulated_makespan_min'] if p[arm]['success'] else 600 for p in full],[p[arm]['condition'] for p in full]),
            'penalty_is_statistical_not_actual_simulation_time':True}
    return {'validated':True,'planned_runs':len(planned),'pilot':plan['pilot'],'raw_steps':steps_total,
        'paired_conditions':len(pairs),'candidate_direct_four_identical_action_pairs':equivalent_candidate_pairs,'groups':groups,'comparisons':comparisons,'clipped_indices_counts':clipping,
        'formal_blind_test':False,'production_acceptance':False,'physical_safety_certified':False,
        'single_executor_only':True,'D_learned_gate_enabled':False,'D_model_domain_gate_enabled':True,
        'limits':'Old disclosed seeds; trusted pre-fault calibration; acknowledged source flows, no unknown flow/model errors. Development controller attribution only.'}


def _uscale(source_root):
    # Read the actual acknowledged coordinate scale used by B adapter, without
    # instantiating a second coupled plant or importing any scoring labels.
    import sys
    sys.path.insert(0,str(Path(source_root)/'vendor/smpl'))
    from smpl.envs.helpers.mab_helpers import uscale
    return np.asarray(uscale)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    result=audit(a.root);write(a.output,result);print(json.dumps(result['groups'],indent=2))
