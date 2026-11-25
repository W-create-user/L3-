"""D role-frozen B05 experiment; train/validation only before test evaluation."""
import argparse,hashlib,json,time,platform,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from l3_system.monitoring.b05_model import B05Model,ENV
from l3_system.monitoring.b05_supervised import B05Supervised

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_episodes(root,manifest,environment,splits):
    episodes=[]
    for e in manifest['episodes']:
        if e['environment_id']!=environment or e['split'] not in splits:continue
        samples=[json.loads(l) for l in (root/e['episode_id']/'samples.jsonl').read_text().splitlines()]
        windows=[]
        for i in range(3,len(samples)):
            windows.append(([s['delivered_observation'] for s in samples[i-3:i+1]],any(s['labels']['sensor_fault_label'] for s in samples[i-3:i+1])))
        episodes.append({**e,'windows':windows})
    return episodes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--supervised-freeze-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.dataset/'manifest.json').read_text());roles={};bindings={}
    for env in ENV:
        rows=[e for e in m['episodes'] if e['environment_id']==env];cal=[];threshold=[]
        for kind in ('clean','bias','noise','missing'):
            group=sorted([e for e in rows if e['split']=='validation' and (e['fault']['kind'] if e['fault'] else 'clean')==kind],key=lambda e:e['episode_id'])
            if len(group)!=2:raise ValueError('two validation lineages per kind required')
            cal.append(group[0]['episode_id']);threshold.append(group[1]['episode_id'])
        roles[env]={'calibration':cal,'threshold':threshold,'train':[e['episode_id'] for e in rows if e['split']=='train']}
        bindings[env]={'environment':env,'dimension':ENV[env][0],'projection_indices':ENV[env][1],
            'collection_config_sha256':m['frozen_config_sha256'],'source_files_sha256':m['source_files_sha256'],
            'profile':m['config']['profiles'][env],'coordinate':'normalize_false source model coordinates'}
    plan={'schema_version':'DB05FitPlan/1.0','roles':roles,'bindings':bindings,'model_version':'supervised-v2' if a.supervised_freeze_only else 'IF-v1','forest_trees':128 if a.supervised_freeze_only else 64,'seed':251125,
        'threshold_policy':'nextafter largest finite negative threshold-validation score, no test tuning',
        'missing_policy':'abstain/block, keep in denominator, no imputation',
        'evaluation_criteria':{'finite_window_recall_min':.8,'finite_window_fpr_max':.05},
        'test_previously_data_audited':True,'formal_D07_blind':False,
        'code_sha256':{n:sha(n) for n in ['l3_system/monitoring/b05_model.py',str(Path(__file__).relative_to(Path.cwd())),'l3_system/monitoring/b05_supervised.py']}}
    write(a.output/'fit_plan.json',plan);models={};receipts={}
    # Both frozen models are written before opening any test observations for predictions.
    for env in ENV:
        data=read_episodes(a.dataset,m,env,{'train','validation'});byid={e['episode_id']:e for e in data}
        rr=roles[env];model=(B05Supervised if a.supervised_freeze_only else B05Model)(env,bindings[env]).fit(*[[byid[eid] for eid in rr[k]] for k in ('train','calibration','threshold')])
        model_path=a.output/(env+'.model.json');digest=model.save(model_path);models[env]=model
        receipts[env]={'path':model_path.name,'sha256':digest,'threshold':model.threshold}
    write(a.output/'model_freeze.json',{'models':receipts,'plan_sha256':sha(a.output/'fit_plan.json'),'before_test_predictions':True})
    if a.supervised_freeze_only:
        print(json.dumps({'frozen':receipts,'test_predictions_performed':False}));return
    results={};window_ledger=[]
    for env,model in models.items():
        counts={'TP':0,'FP':0,'FN':0,'TN':0};unavailable=0;test=read_episodes(a.dataset,m,env,{'test'});per_episode=[]
        for e in test:
            alarms=missing=0
            for i,(w,label) in enumerate(e['windows'],4):
                result=model.score_window(w)
                if result['status']=='unavailable':unavailable+=1;missing+=1
                else:
                    key=('T' if result['anomalous']==label else 'F')+('P' if result['anomalous'] else 'N');counts[key]+=1;alarms+=int(result['anomalous'])
                window_ledger.append({'episode_id':e['episode_id'],'environment':env,'end_record':i,'synthetic_label':label,'result':result})
            per_episode.append({'episode_id':e['episode_id'],'windows':len(e['windows']),'unavailable':missing,'alarms':alarms,'has_injected_fault':e['fault'] is not None})
        recall=counts['TP']/(counts['TP']+counts['FN']) if counts['TP']+counts['FN'] else None
        fpr=counts['FP']/(counts['FP']+counts['TN']) if counts['FP']+counts['TN'] else None
        results[env]={'test_source_episodes':len(test),'finite_window_counts':counts,'missing_windows_blocked':unavailable,
            'all_planned_windows':sum(len(e['windows']) for e in test),'finite_recall':recall,'finite_fpr':fpr,
            'numerical_criteria_met':recall is not None and recall>=.8 and fpr is not None and fpr<=.05,
            'per_episode':per_episode,'deployment_authorized':False}
    write(a.output/'test_window_ledger.json',window_ledger)
    write(a.output/'results.json',{'schema_version':'DB05Evaluation/1.0','models':receipts,'results':results,
        'test_used_to_tune':False,'physical_safety_certified':False,'formal_blind_test':False})
    print(json.dumps(results))
if __name__=='__main__':main()
