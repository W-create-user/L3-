"""Reconcile D research evidence without turning software checks into acceptance.

This is an evidence audit, not an independent reviewer signature or deployment gate.
"""
import hashlib
import json

COUNT_KEYS=('TP','FP','FN','TN')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(original,reproduced,fusion):
    if original!=reproduced:raise ValueError('frozen sensor report did not reproduce exactly')
    if original.get('label_scope')!='sensor_fault' or original.get('physical_process_labels_available') is not False:
        raise ValueError('reviewed sensor-only evidence required')
    splits={}
    for name in ('validation','test'):
        part=original['evaluation'][name];totals={key:0 for key in COUNT_KEYS};ids=set()
        events=detected=missed=unscorable=0;rates=[]
        for row in part['episodes']:
            if row['episode_id'] in ids:raise ValueError('duplicate evaluation episode')
            ids.add(row['episode_id']);counts=row['confusion']
            for key in COUNT_KEYS:
                if type(counts[key])is not int or counts[key]<0:raise ValueError('nonnegative integer counts required')
                totals[key]+=counts[key]
            if sum(counts[key] for key in COUNT_KEYS)!=row['windows']:raise ValueError('window denominator mismatch')
            normal=counts['FP']+counts['TN'];rates.append(counts['FP']/normal if normal else None)
            ev=row['event_evaluation']
            events+=ev['event_count'];detected+=ev['detected_event_count'];missed+=ev['missed_event_count'];unscorable+=ev['unscorable_event_count']
        if totals!=part['counts']:raise ValueError('aggregate confusion differs from episodes')
        if detected+missed+unscorable!=events:raise ValueError('event denominator mismatch')
        summary=part['event_summary']
        if any(summary[k]!=v for k,v in {'episode_count':len(ids),'event_count':events,'detected_event_count':detected,'missed_event_count':missed,'unscorable_event_count':unscorable,'window_count':sum(totals.values())}.items()):raise ValueError('event summary mismatch')
        normal=totals['FP']+totals['TN'];positive=totals['TP']+totals['FN']
        splits[name]={'episodes':len(ids),'window_counts':totals,'normal_window_denominator':normal,
            'window_false_positive_rate':totals['FP']/normal if normal else None,
            'window_recall':totals['TP']/positive if positive else None,
            'episode_normal_window_false_positive_rates':rates,'raw_events':events,'detected_raw_events':detected,
            'missed_raw_events':missed,'unscorable_raw_events':unscorable,
            'whole_uncorrupted_episodes':summary['all_normal_episode_count'],
            'whole_normal_episode_false_alarm_rate':None if not summary['all_normal_episode_count'] else summary['all_normal_episode_with_alarm_count']/summary['all_normal_episode_count'],
            'population_confidence_interval':None,'uncertainty_reason':'Few declared source episodes; statistical independence and operating coverage unverified. Overlapping windows are not independent trials.'}
    if fusion.get('schema_version')!='D04DerivedSensorStudy/1.0' or fusion.get('independent_research_acceptance_passed') is not False:raise ValueError('exploratory derived study required')
    if fusion.get('original_inputs_unchanged') is not True:raise ValueError('source preservation evidence required')
    receipts=[v['private_artifact_receipt'] for r in fusion['runs'] for v in r['variants'].values()]
    if len(receipts)!=21 or not all(r['validation_roundtrip_exact'] is True for r in receipts):raise ValueError('all 21 private checkpoints must roundtrip')
    statuses=[
        {'item':'frozen sensor baseline reproducibility','status':'verified','evidence':'exact report equality; private model hash comparison separately recorded'},
        {'item':'derived three-branch training and checkpoint restoration','status':'engineering_verified','evidence':'21 independently trained models; exact validation logit roundtrips'},
        {'item':'independent sensor-fault research acceptance','status':'evidence_insufficient','missing':'untouched independent source groups, whole-normal validation/test episodes, prespecified acceptance criteria and one final frozen evaluation'},
        {'item':'physical process anomaly detection','status':'evidence_insufficient','missing':'physically justified anomaly labels and operating envelope; sensor bias is not physical truth'},
        {'item':'independent visual/text sensing capability','status':'not_evaluated','missing':'causal independent camera/log data; numeric-derived modalities cannot establish this capability'},
        {'item':'mAb physical R-03','status':'not_evaluated','missing':'separate confirmed variable units, operating conditions, threshold sources and labelled cases'},
        {'item':'project success >=90% and makespan reduction >=20%','status':'not_evaluated','missing':'frozen physical goal/deadline definitions and matched started-run ledger with all failures'},
        {'item':'independent final acceptance','status':'not_signed','missing':'separate reviewer reproduction and final evidence sign-off'}]
    return {'schema_version':'DResearchEvidenceAudit/1.0','sensor_evaluation':splits,
        'checkpoints_verified':len(receipts),'acceptance_items':statuses,'all_research_requirements_passed':False,
        'deployment_authorized':False,'thresholds_changed':False,'test_used_for_tuning':False,
        'limits':'Reproduction reuses previously inspected data. This report records evidence gaps; it neither invents physical thresholds nor certifies research success.'}


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','reproduced','fusion','original-model','reproduced-model','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit(*(json.loads(getattr(args,n).read_text()) for n in ('original','reproduced','fusion')))
    first=digest(args.original_model);second=digest(args.reproduced_model)
    if first!=second:raise ValueError('frozen model bytes did not reproduce')
    result['frozen_model_sha256']=first
    result['frozen_model_bytes_reproduced_identically']=True
    result['input_sha256']={n:digest(getattr(args,n)) for n in ('original','reproduced','fusion')}
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Evidence audit complete; unmet research acceptance items remain explicit.')
