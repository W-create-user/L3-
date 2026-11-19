"""Episode-paired descriptive results and lineage-cluster bootstrap.

Evidence declarations do not verify goal definitions or statistical independence.
"""
import numpy as np


def analyze(rows,*,seed=1511,resamples=2000):
    if type(seed)is not int or seed<0 or type(resamples)is not int or not 100<=resamples<=100000:
        raise ValueError('integer seed and 100..100000 bootstrap samples required')
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty paired episode records required')
    pairs={};group_condition={}
    fields={'pair_id','group_id','condition','arm','success','simulated_makespan','evidence_ref'}
    for row in rows:
        if not isinstance(row,dict) or set(row)!=fields:raise ValueError('exact episode fields required')
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ('pair_id','group_id','condition','evidence_ref')):
            raise ValueError('pair, lineage, condition and evidence references required')
        if row['arm'] not in ('baseline','candidate') or type(row['success'])is not bool:
            raise ValueError('explicit arm and success required; failures must not be excluded')
        value=row['simulated_makespan']
        if value is not None and (type(value) not in (int,float) or not np.isfinite(value) or value<=0):
            raise ValueError('positive simulated makespan or explicit null required')
        pair=pairs.setdefault(row['pair_id'],{})
        if row['arm'] in pair:raise ValueError('duplicate episode arm in pair')
        if pair:
            prior=next(iter(pair.values()))
            if any(prior[k]!=row[k] for k in ('group_id','condition')):raise ValueError('pair conditions or lineage differ')
        pair[row['arm']]=row
        group=row['group_id'];condition=row['condition']
        if group in group_condition and group_condition[group]!=condition:
            raise ValueError('shared lineage across conditions needs a different resampling design')
        group_condition[group]=condition
    if any(set(pair)!={'baseline','candidate'} for pair in pairs.values()):raise ValueError('incomplete pairs cannot be silently dropped')
    clusters={};baseline_success=candidate_success=0;time_pairs=[]
    for pair in pairs.values():
        b,c=pair['baseline'],pair['candidate']
        baseline_success+=b['success'];candidate_success+=c['success']
        clusters.setdefault(c['condition'],{}).setdefault(c['group_id'],[]).append(int(c['success'])-int(b['success']))
        if b['success'] and c['success'] and b['simulated_makespan'] is not None and c['simulated_makespan'] is not None:
            time_pairs.append((b['simulated_makespan']-c['simulated_makespan'])/b['simulated_makespan'])
    # Resample complete lineage groups within each condition; every selected group
    # contributes all its episode pairs. Never resample overlapping windows.
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        values=[]
        for condition in sorted(clusters):
            groups=clusters[condition]
            arrays=[groups[key] for key in sorted(groups)]
            for index in rng.integers(0,len(arrays),len(arrays)):values.extend(arrays[index])
        draws.append(float(np.mean(values)))
    n=len(pairs)
    return {'schema_version':'D08PairedAnalysis/0.1','paired_episode_count':n,'lineage_group_count':len(group_condition),
        'baseline_success_count':baseline_success,'candidate_success_count':candidate_success,
        'baseline_success_rate':baseline_success/n,'candidate_success_rate':candidate_success/n,
        'success_rate_difference':(candidate_success-baseline_success)/n,
        'success_difference_percentile_95_interval':np.quantile(draws,[.025,.975]).tolist(),
        'resampling_unit':'lineage group, stratified by condition; complete matched pairs retained',
        'seed':seed,'resamples':resamples,'complete_successful_time_pair_count':len(time_pairs),
        'time_pair_exclusion_count':n-len(time_pairs),
        'mean_relative_makespan_reduction_complete_success_pairs':float(np.mean(time_pairs)) if time_pairs else None,
        'makespan_comparison_scope':'descriptive successful complete cases only; survivor selection, no overall efficiency claim',
        'small_cluster_warning':any(len(groups)<10 for groups in clusters.values()),
        'evidence_claims_verified':False,'target_achievement_claimed':False,
        'limits':'All supplied started pairs count for success. Corrupt evidence/excluded runs need a separate original-started ledger. References and physical goals are not verified. Small numbers of lineage groups make bootstrap intervals unreliable.'}


if __name__=='__main__':
    import argparse,json,hashlib
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1511);parser.add_argument('--resamples',type=int,default=2000)
    args=parser.parse_args()
    raw=args.records.read_bytes()
    def invalid(value):raise ValueError('nonfinite JSON: '+value)
    rows=json.loads(raw,parse_constant=invalid)
    result=analyze(rows,seed=args.seed,resamples=args.resamples)
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    with args.output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
