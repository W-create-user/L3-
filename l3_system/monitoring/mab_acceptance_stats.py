"""A08 ratio-of-paired-means statistics; retain failures and planned denominator."""
import math
import numpy as np

def wilson(success,total):
    if total==0:return None
    z=1.959963984540054;p=success/total;d=1+z*z/total
    c=(p+z*z/(2*total))/d;w=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,c-w),min(1,c+w)]

def analyze(plan,rows,*,seed=251125,resamples=10000,penalty_minutes=600.,wall_budget_seconds=90.):
    if not isinstance(plan,list) or not plan or type(resamples)is not int or resamples<100:raise ValueError('planned rows and >=100 bootstrap draws required')
    expected={r['run_id']:r for r in plan}
    if len(expected)!=len(plan):raise ValueError('duplicate planned run')
    recorded={r['run_id']:r for r in rows}
    if len(recorded)!=len(rows) or not set(recorded)<=set(expected):raise ValueError('duplicate/unplanned results')
    pairs={};arms={}
    for ident,spec in expected.items():
        if spec['arm'] not in ('baseline','candidate'):raise ValueError('two-arm statistics only')
        # Missing receipt does not invent a not-started classification.
        row=recorded.get(ident,{'run_id':ident,'started':None,'success':False,'simulated_makespan_min':None,'failure_reason':'missing_receipt'})
        if (row.get('started') is not None and type(row.get('started'))is not bool) or type(row.get('success'))is not bool:raise ValueError('explicit start/success states required')
        if any(k in row and row[k]!=spec[k] for k in ('pair_id','condition','arm')):raise ValueError('result/spec mismatch')
        value=row.get('simulated_makespan_min')
        if row['success'] and (row['started'] is not True or type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=penalty_minutes):raise ValueError('success requires verified start and finite on-time cycle')
        wall=row.get('wall_seconds')
        if row['success'] and (type(wall) not in (int,float) or not math.isfinite(wall) or not 0<=wall<=wall_budget_seconds):raise ValueError('success requires verified within-budget wall time')
        r={**row,**spec};pair=pairs.setdefault(spec['pair_id'],{})
        if spec['arm'] in pair:raise ValueError('duplicate arm')
        pair[spec['arm']]=r;arms.setdefault(spec['arm'],[]).append(r)
    if any(set(p)!= {'baseline','candidate'} for p in pairs.values()):raise ValueError('complete planned pairs required')
    for pair in pairs.values():
        b,c=pair['baseline'],pair['candidate']
        if b['condition']!=c['condition'] or b['group_id']!=c['group_id']:raise ValueError('pair lineage/condition mismatch')
    def summary(rr):
        n=len(rr);s=sum(r['success'] for r in rr);known=sum(r['started'] is True for r in rr);unknown=sum(r['started'] is None for r in rr)
        return {'planned':n,'verified_started':known,'unknown_start':unknown,'not_started':n-known-unknown,'successes':s,
            'verified_started_failures':sum(r['started'] is True and not r['success'] for r in rr),
            'success_over_verified_started':s/known if known else None,'wilson95_verified_started':wilson(s,known),
            'success_over_planned':s/n,'wilson95_planned':wilson(s,n),'conservative_started_rate':s/(known+unknown) if known+unknown else None}
    by_condition={cond:{arm:summary([r for r in rr if r['condition']==cond]) for arm,rr in arms.items()} for cond in sorted({s['condition'] for s in plan})}
    def times(ps,complete):
        b=[];c=[]
        for pair in ps:
            rb,rc=pair['baseline'],pair['candidate']
            if complete and not (rb['success'] and rc['success']):continue
            b.append(rb['simulated_makespan_min'] if rb['success'] else penalty_minutes)
            c.append(rc['simulated_makespan_min'] if rc['success'] else penalty_minutes)
        return None if not b else {'pairs':len(b),'baseline_mean_minutes':float(np.mean(b)),'candidate_mean_minutes':float(np.mean(c)),
            'improvement':float(1-np.mean(c)/np.mean(b))}
    clusters={};owners={}
    for pair in pairs.values():
        r=pair['baseline'];group=r['group_id'];cond=r['condition']
        if group in owners and owners[group]!=cond:raise ValueError('lineage across conditions requires a different design')
        owners[group]=cond;clusters.setdefault(cond,{}).setdefault(group,[]).append(pair)
    rng=np.random.default_rng(seed);draws={False:[],True:[]}
    for _ in range(resamples):
        ps=[]
        for cond,groups in sorted(clusters.items()):
            values=list(groups.values())
            for i in rng.integers(0,len(values),len(values)):ps.extend(values[i])
        for complete in draws:
            result=times(ps,complete)
            if result is not None:draws[complete].append(result['improvement'])
    reports={}
    for name,complete in [('common_success',True),('all_planned_600min_sensitivity',False)]:
        result=times(list(pairs.values()),complete);reports[name]={'point':result,'bootstrap95':np.quantile(draws[complete],[.025,.975]).tolist() if draws[complete] else None,
            'valid_bootstrap_draws':len(draws[complete]),'degenerate_interval':bool(draws[complete]) and min(draws[complete])==max(draws[complete]),
            'penalty_is_actual_time':False}
    return {'schema_version':'DMAbAcceptanceStats/1.0','arms':{a:summary(rr) for a,rr in arms.items()},'by_condition':by_condition,
        'efficiency':reports,'failure_details':[{k:r.get(k) for k in ('run_id','started','failure_reason')} for rr in arms.values() for r in rr if not r['success']],
        'resampling':'complete source-lineage clusters stratified by condition','seed':seed,'resamples':resamples,
        'small_cluster_warning':any(len(g)<10 for g in clusters.values()),'physical_safety_certified':False,
        'target_pass_automatically_claimed':False}
