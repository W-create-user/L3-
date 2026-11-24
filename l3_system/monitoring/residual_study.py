"""Frozen fresh holdout evaluation. Never selects a model on the fresh test."""
from contextlib import redirect_stdout
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from .residual_detector import ResidualDetector
from .anomaly_detector import AnomalyDetector

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(frozen,baseline,output,*,seed_base=28001):
    frozen=Path(frozen);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    record=json.loads((frozen/'freeze.json').read_text())
    model=ResidualDetector.load_trusted(frozen/'model.pkl',record['model_sha256'])
    old=AnomalyDetector.load_trusted(baseline)
    recipes=[None]*6+[(8,15,1.05),(17,28,1.10),(30,44,1.20),(10,19,.90),(23,40,.80),(39,53,1.15)]
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':record['model_sha256'],
          'baseline_model_sha256':sha(Path(baseline)/'model.pkl'),
          'source_sha256':sha(__file__),'model_source_sha256':sha(Path(__file__).with_name('residual_detector.py')),
          'acceptance_scope':'predeclared engineering sensor-bias criteria on this small fresh holdout, no physical/generalization certification',
          'criteria':{'window_recall_min':.8,'normal_window_fpr_max':.05,'whole_normal_episode_fpr_max':.1,'raw_event_recall_min':.8},
          'development_history':'v1 holdout (28001..28012) informed the threshold tie-break correction; v2 uses new 38001..38012 holdout. Keep both results.',
          'episodes':[{'episode_id':f'holdout_{seed_base+i}','seed':seed_base+i,'horizon':60,'fault':r} for i,r in enumerate(recipes)]}
    (output/'preregistered_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    # Persist all choices before constructing any new simulator or reading test.
    from l3_system.execution import EnvironmentAdapter,ExecutionAgent
    from l3_system.agents.monitoring_agent import MonitoringAgent
    episodes=[];initials=set()
    for spec in plan['episodes']:
        folder=output/spec['episode_id'];folder.mkdir()
        e={'episode_id':spec['episode_id'],'split':'test','lineage_group':spec['episode_id'],
           'label_scope':'sensor_fault','label_source':'predeclared sensor-copy intervention','observations':[],'labels':[]}
        with (folder/'stdout.txt').open('w') as log,redirect_stdout(log),EnvironmentAdapter(max_steps=65) as adapter:
            ex=ExecutionAgent(adapter,log_path=folder/'B.jsonl');ex.reset(seed=spec['seed'])
            initial=hashlib.sha256(np.asarray(adapter._env.xk.full(),dtype=np.float64).tobytes()).hexdigest()
            if initial in initials:raise ValueError('duplicated holdout initial state')
            initials.add(initial);e['initial_internal_state_sha256']=initial
            monitor=MonitoringAgent(adapter.action_space,adapter.observation_space)
            with (folder/'trajectory.jsonl').open('w') as stream:
                for step in range(1,61):
                    action=(adapter.action_space.low+adapter.action_space.high)/2
                    result=ex.execute_assignment({'task_id':f"{spec['episode_id']}-{step}",'task_type':'control_cycle'},monitor.check_request,action=action)
                    if not result['success'] or not result['executed'] or result['done']:raise RuntimeError('partial holdout; retain failure')
                    raw=list(result['observation']);delivered=list(raw);fault=spec['fault']
                    active=bool(fault and fault[0]<=step<=fault[1])
                    if active:delivered[6]*=fault[2]
                    features=list(delivered);features[0]=0.
                    e['observations'].append(features);e['labels'].append(active)
                    stream.write(json.dumps({'step':step,'raw':raw,'delivered':delivered,'sensor_bias':active,
                        'physical_health':None,'action':result['action'],'state_version':adapter.state_version},allow_nan=False)+'\n')
                    stream.flush()
        (folder/'model_episode.json').write_text(json.dumps(e,allow_nan=False)+'\n');episodes.append(e)
        print(spec['episode_id'],'collected',flush=True)
    result=model.evaluate(episodes)
    baseline_rows=[]
    for e in episodes:
        x,y=old._windows(e);pred=-old.model.score_samples(old.scaler.transform(x))>=old.threshold
        from .anomaly_detector import confusion
        baseline_rows.append({'episode_id':e['episode_id'],'counts':confusion(y,pred),
            'whole_normal':not any(e['labels']),'any_alarm':bool(pred.any())})
    counts=result['counts'];recall=counts['TP']/(counts['TP']+counts['FN']);fpr=counts['FP']/(counts['FP']+counts['TN'])
    whole=result['whole_normal_with_alarm']/result['whole_normal_episodes']
    event_recall=result['events']['scorable_event_recall']
    c=plan['criteria'];passed=(recall>=c['window_recall_min'] and fpr<=c['normal_window_fpr_max'] and whole<=c['whole_normal_episode_fpr_max'] and event_recall>=c['raw_event_recall_min'])
    report={'schema_version':'DResidualHoldout/1.0','plan_sha256':sha(output/'preregistered_plan.json'),
        'frozen_model_sha256':record['model_sha256'],'initial_state_hashes_unique':len(initials),'steps':720,
        'test_evaluations':1,'test_used_for_selection':False,'result':result,
        'baseline':{'episodes':baseline_rows,'counts':{k:sum(r['counts'][k] for r in baseline_rows) for k in ('TP','FP','FN','TN')}},
        'engineering_criteria_passed':bool(passed),'window_recall':recall,'normal_window_fpr':fpr,'whole_normal_episode_fpr':whole,
        'physical_safety_certified':False,'production_metrics_evaluated':False,'population_independence_established':False}
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('engineering_criteria_passed','window_recall','normal_window_fpr','whole_normal_episode_fpr')}))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-base',type=int,default=28001)
    a=p.parse_args();run(a.frozen,a.baseline,a.output,seed_base=a.seed_base)
