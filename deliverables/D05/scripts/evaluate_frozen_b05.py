"""Evaluate frozen model once; keep missing windows and every failure visible."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from l3_system.monitoring.b05_model import B05Model
from deliverables.D05.scripts.fit_b05_models import read_episodes,write,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.dataset/'manifest.json').read_text());freeze=json.loads((a.models/'model_freeze.json').read_text());results={};ledger=[]
    if not m['complete']:raise ValueError('incomplete held-out collection')
    if m['schema_version']=='DNewSensorHoldout/1.0':
        plan=json.loads((a.dataset/'collection_plan.json').read_text())
        if plan['model_freeze_sha256']!=sha(a.models/'model_freeze.json'):raise ValueError('model freeze changed')
    for env,f in freeze['models'].items():
        model=B05Model.load_trusted(a.models/f['path'],f['sha256']);counts={'TP':0,'FP':0,'FN':0,'TN':0};missing=0;episodes=[]
        for e in read_episodes(a.dataset,m,env,{'test'}):
            local={'episode_id':e['episode_id'],'windows':len(e['windows']),'missing':0,'alarms':0,'source_fault':None if e['fault'] is None else e['fault']['kind']}
            for i,(w,y) in enumerate(e['windows'],4):
                result=model.score_window(w)
                if result['status']=='unavailable':missing+=1;local['missing']+=1
                else:
                    pred=result['anomalous'];counts[('T' if pred==y else 'F')+('P' if pred else 'N')]+=1;local['alarms']+=int(pred)
                ledger.append({'environment':env,'episode_id':e['episode_id'],'end_record':i,'synthetic_label':y,'result':result})
            episodes.append(local)
        recall=counts['TP']/(counts['TP']+counts['FN']);fpr=counts['FP']/(counts['FP']+counts['TN'])
        clean=[e for e in episodes if e['source_fault'] is None];clean_alarm_rate=sum(e['alarms']>0 for e in clean)/len(clean)
        results[env]={'whole_clean_episode_alarm_rate':clean_alarm_rate,'finite_counts':counts,'finite_recall':recall,'finite_fpr':fpr,'missing_windows_blocked':missing,'planned_windows':sum(e['windows'] for e in episodes),'episodes':episodes,'numerical_criteria_met':recall>=.8 and fpr<=.05 and clean_alarm_rate<=.1,'deployment_authorized':False}
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'results.json',{'models':freeze['models'],'results':results,'test_used_for_tuning':False,'formal_D07_blind':False,'physical_safety_certified':False});write(a.output/'window_ledger.json',ledger);print(json.dumps(results))
if __name__=='__main__':main()
