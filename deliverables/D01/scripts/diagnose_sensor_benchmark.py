"""Frozen-model diagnostics and shadow candidate counts; no tuning or execution.

Already inspected validation/test data: diagnostics are exploratory, not a new test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_r03_supplement import review,candidate_status,SOURCE_INDICES,safe_path
from l3_system.monitoring.anomaly_detector import AnomalyDetector,features,confusion


def phase_summary(labels,predictions,scores,window_size):
    if (type(window_size)is not int or window_size<2 or len(labels)<window_size
            or any(type(x)is not bool for x in labels)
            or len(predictions)!=len(labels)-window_size+1 or len(scores)!=len(predictions)
            or any(type(x)is not bool for x in predictions) or not np.isfinite(scores).all()):
        raise ValueError('complete windows, boolean labels/predictions and finite scores required')
    positives=[i for i,value in enumerate(labels) if value]
    groups={name:[] for name in ('uncorrupted_reference','before_injection','injection','post_injection_window_tail','recovered_window')}
    for j,end in enumerate(range(window_size-1,len(labels))):
        truth=any(labels[end-window_size+1:end+1])
        if not positives:name='uncorrupted_reference'
        elif labels[end]:name='injection'
        elif truth:name='post_injection_window_tail'
        elif end<positives[0]:name='before_injection'
        else:name='recovered_window'
        groups[name].append((truth,predictions[j],scores[j]))
    result={}
    for name,values in groups.items():
        truth=[v[0] for v in values];pred=[v[1] for v in values]
        result[name]={'windows':len(values),'alarms':sum(pred),'confusion':confusion(truth,pred),
            'mean_anomaly_score':float(np.mean([v[2] for v in values])) if values else None}
    return {'warmup_records_excluded':window_size-1,'phases':result}


def diagnose(episodes,detector):
    if detector.label_scope!='sensor_fault' or detector.dimension!=len(SOURCE_INDICES):
        raise ValueError('only the reviewed sensor-fault projection model is supported')
    w=detector.window_size;out={};false_windows=[]
    for split in ('train','validation','test'):
        reports=[]
        for episode in [e for e in episodes if e['split']==split]:
            data=np.asarray(episode['observations'],dtype=float);labels=episode['labels']
            windows=[data[end-w+1:end+1] for end in range(w-1,len(data))]
            values=-detector.model.score_samples(detector.scaler.transform(np.asarray([features(window) for window in windows])))
            pred=[bool(value>=detector.threshold) for value in values]
            reports.append({'episode_id':episode['episode_id'],**phase_summary(labels,pred,values.tolist(),w)})
            if split!='train':
                false_windows.extend(window for end,window,alarm in zip(range(w-1,len(data)),windows,pred)
                                     if alarm and not any(labels[end-w+1:end+1]))
        phases={}
        for name in reports[0]['phases']:
            entries=[r['phases'][name] for r in reports];n=sum(e['windows'] for e in entries)
            phases[name]={'windows':n,'alarms':sum(e['alarms'] for e in entries),
                'confusion':{k:sum(e['confusion'][k] for e in entries) for k in ('TP','FP','FN','TN')},
                'mean_anomaly_score':sum(e['mean_anomaly_score']*e['windows'] for e in entries if e['windows'])/n if n else None}
        out[split]={'episode_count':len(reports),'warmup_records_excluded':len(reports)*(w-1),'phases':phases}
    medians=np.median(np.concatenate([e['observations'] for e in episodes if e['split']=='train']),axis=0)
    sensitivities=[]
    if false_windows:
        originals=np.asarray([features(window) for window in false_windows])
        original_scores=-detector.model.score_samples(detector.scaler.transform(originals))
        for column,index in enumerate(SOURCE_INDICES):
            replaced=[]
            for window in false_windows:
                changed=window.copy();changed[:,column]=medians[column];replaced.append(features(changed))
            changed_scores=-detector.model.score_samples(detector.scaler.transform(np.asarray(replaced)))
            sensitivities.append({'source_observation_index':index,'mean_score_reduction':float(np.mean(original_scores-changed_scores)),
                'alarms_cleared':int(np.sum(changed_scores<detector.threshold)),'false_alarm_windows_examined':len(false_windows)})
    return {'phases':out,'median_replacement_sensitivity':sorted(sensitivities,key=lambda r:r['mean_score_reduction'],reverse=True),
        'sensitivity_limits':'Train-only raw-variable medians replace one full-window column at a time. Off-manifold interventions; not causal attribution, feature retraining or physical control suggestions.',
        'threshold_unchanged':detector.threshold,'retrained':False,'test_status':'previously inspected; exploratory diagnosis only'}


def shadow(package):
    package=Path(package).resolve();data=json.loads((package/'sensor_benchmark/episodes.json').read_text())
    result={split:{'step_records':0,'sensor_positive_records':0,'status_disagreements':0,
        'environment_observation':{},'delivered_observation':{}} for split in ('train','validation','test')}
    for episode in data['episodes']:
        counts=result[episode['split']]
        path=safe_path(package/'sensor_benchmark',episode['raw_trajectory'])
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['step']==0:continue
            counts['step_records']+=1;counts['sensor_positive_records']+=row['sensor_fault_label']
            statuses=[]
            for field in ('environment_observation','delivered_observation'):
                status=candidate_status(row[field]);statuses.append(status)
                counts[field][status]=counts[field].get(status,0)+1
            counts['status_disagreements']+=int(statuses[0]!=statuses[1])
    return {'candidate_relative_limit':.10,'counts':result,'runtime_enabled':False,'operational_decision':None,
        'physical_labels_available':False,'limits':'Raw environment is an audit comparison, never an IF feature. Candidate hits are not physical hazard labels; no physical precision/recall is computable.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,required=True);parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--expected-model-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    episodes,receipt=review(args.package_root,args.evidence_root)
    actual=hashlib.sha256((args.model_dir/'model.pkl').read_bytes()).hexdigest()
    if actual!=args.expected_model_sha256:raise ValueError('trusted training artifact fingerprint mismatch; do not unpickle')
    detector=AnomalyDetector.load_trusted(args.model_dir)
    if detector.card['intake_receipt']!=receipt:raise ValueError('training/source receipt mismatch')
    result={'schema_version':'D03ExploratoryDiagnostics/0.1','source_receipt':receipt,'model_sha256':actual,
        'diagnostics':diagnose(episodes,detector),'shadow':shadow(args.package_root),'deployment_authorized':False}
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Completed frozen sensor-model diagnostics and disabled candidate shadow evaluation; no simulator actions.')

if __name__=='__main__':main()
