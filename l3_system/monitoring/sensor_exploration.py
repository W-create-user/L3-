"""Train/validation-only sensor experiments. Test input is explicitly forbidden.

No simulator execution, deployment artifacts or physical-health claims.
"""
import hashlib
import json
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .anomaly_detector import features,confusion

BLOCKS={'mean':0,'std':1,'min':2,'max':3,'delta':4,'absdiff':5}


def validate_config(config):
    fields={'schema_version','window_sizes','feature_sets','projections','normal_window_fpr_budget','seed','trees'}
    if not isinstance(config,dict) or set(config)!=fields or config['schema_version']!='SensorExploration/0.1':
        raise ValueError('explicit exploration configuration required')
    windows=config['window_sizes']
    if not isinstance(windows,list) or not windows or len(windows)>4 or any(type(w)is not int or not 2<=w<=60 for w in windows) or len(set(windows))!=len(windows):
        raise ValueError('1..4 distinct windows in 2..60 required')
    if type(config['seed'])is not int or config['seed']<0 or type(config['trees'])is not int or not 10<=config['trees']<=200:
        raise ValueError('explicit seed and bounded forest budget required')
    budget=config['normal_window_fpr_budget']
    if type(budget) not in (int,float) or not np.isfinite(budget) or not 0<=budget<=1:raise ValueError('FPR budget must be in [0,1]')
    sets=config['feature_sets'];projections=config['projections']
    if not isinstance(sets,dict) or not 1<=len(sets)<=4 or not isinstance(projections,dict) or not 1<=len(projections)<=4:
        raise ValueError('bounded feature/projection grid required')
    for name,groups in sets.items():
        if (not isinstance(name,str) or not name or not isinstance(groups,list) or not groups
                or any(not isinstance(g,str) or g not in BLOCKS for g in groups) or len(set(groups))!=len(groups)):
            raise ValueError('distinct known statistical feature groups required')
    for name,indices in projections.items():
        if (not isinstance(name,str) or not name or not isinstance(indices,list) or not indices
                or any(type(i)is not int or not 1<=i<=38 for i in indices) or len(set(indices))!=len(indices)):
            raise ValueError('observation indices 1..38 only; time coordinate 0 forbidden')


def _episodes(episodes,largest_window):
    if not isinstance(episodes,list) or not episodes:raise ValueError('nonempty episodes required')
    split_rows={'train':[],'validation':[]};ids=set();groups={};signatures={};scope=None
    for episode in episodes:
        if not isinstance(episode,dict):raise ValueError('episode objects required')
        split=episode.get('split')
        if split=='test':raise ValueError('test data forbidden in exploration, even without labels')
        if split not in split_rows:raise ValueError('only train/validation splits accepted')
        eid=episode.get('episode_id');group=episode.get('lineage_group')
        if not isinstance(eid,str) or not eid or eid in ids or not isinstance(group,str) or not group:raise ValueError('unique IDs and lineage required')
        if group in groups and groups[group]!=split:raise ValueError('lineage crosses train/validation')
        label_scope=episode.get('label_scope')
        if label_scope not in ('sensor_fault','synthetic_fixture') or scope is not None and scope!=label_scope:raise ValueError('one sensor/fixture scope required; not physical health')
        if not isinstance(episode.get('label_source'),str) or not episode['label_source'].strip():raise ValueError('explicit label provenance required')
        values=np.asarray(episode['observations']);labels=episode['labels']
        if (values.dtype.kind not in 'fiu' or values.ndim!=2 or values.shape[1]!=39 or len(values)<largest_window
                or not np.isfinite(values).all() or not isinstance(labels,list) or len(labels)!=len(values)
                or any(type(v)is not bool for v in labels)):raise ValueError('complete finite raw39 episodes with explicit boolean labels required')
        if split=='train' and any(labels):raise ValueError('training must be uncorrupted for the declared sensor scope')
        signature=hashlib.sha256(json.dumps(values[:,1:].astype(float).tolist(),allow_nan=False).encode()).hexdigest()
        if signature in signatures and signatures[signature]!=split:raise ValueError('identical non-clock trajectory crosses splits')
        signatures[signature]=split;groups[group]=split;ids.add(eid);scope=label_scope
        split_rows[split].append(episode)
    if not all(split_rows.values()):raise ValueError('train and validation episodes required')
    if not any(not any(e['labels']) for e in split_rows['validation']):raise ValueError('at least one whole uncorrupted validation episode required')
    if not any(any(e['labels']) for e in split_rows['validation']):raise ValueError('independent sensor-positive validation episode required')
    return split_rows,scope


def _windows(episode,w,indices,blocks):
    data=np.asarray(episode['observations'],dtype=float)[:,indices];width=len(indices)
    select=[BLOCKS[group]*width+i for group in blocks for i in range(width)]
    x=np.asarray([features(data[end-w+1:end+1])[select] for end in range(w-1,len(data))])
    y=np.asarray([any(episode['labels'][end-w+1:end+1]) for end in range(w-1,len(data))],dtype=bool)
    return x,y


def select_threshold(labels,scores,budget):
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=float)
    if labels.shape!=scores.shape or labels.ndim!=1 or not np.isfinite(scores).all() or not labels.any() or labels.all():
        raise ValueError('validation needs finite scores and both classes')
    choices=np.unique(np.r_[scores,np.nextafter(scores.max(),np.inf)])
    feasible=[]
    for threshold in choices:
        counts=confusion(labels,scores>=threshold)
        if counts['false_positive_rate']<=budget:
            feasible.append((counts['recall'],counts['precision'] or 0,float(threshold),counts))
    chosen=max(feasible,key=lambda item:item[:3])
    return chosen[2],chosen[3]


def explore(episodes,config):
    validate_config(config);splits,scope=_episodes(episodes,max(config['window_sizes']));results=[]
    for window in config['window_sizes']:
        for projection,indices in config['projections'].items():
            for name,blocks in config['feature_sets'].items():
                train=np.concatenate([_windows(e,window,indices,blocks)[0] for e in splits['train']])
                validation=[(e,*_windows(e,window,indices,blocks)) for e in splits['validation']]
                vx=np.concatenate([x for _,x,_ in validation]);vy=np.concatenate([y for _,_,y in validation])
                scaler=StandardScaler().fit(train)
                model=IsolationForest(n_estimators=config['trees'],random_state=config['seed'],contamination='auto',n_jobs=1)
                model.fit(scaler.transform(train));scores=-model.score_samples(scaler.transform(vx))
                threshold,counts=select_threshold(vy,scores,config['normal_window_fpr_budget'])
                offset=0;whole_normal_count=whole_normal_alarm_count=0
                for e,x,y in validation:
                    pred=scores[offset:offset+len(x)]>=threshold;offset+=len(x)
                    if not any(e['labels']):whole_normal_count+=1;whole_normal_alarm_count+=int(pred.any())
                results.append({'window_size':window,'projection':projection,'source_indices':indices,'feature_set':name,
                    'feature_blocks':blocks,'feature_dimension':train.shape[1],'train_windows':len(train),
                    'validation_windows':len(vx),'threshold':threshold,'validation':counts,
                    'whole_uncorrupted_validation_episodes':whole_normal_count,
                    'whole_uncorrupted_validation_episodes_with_alarm':whole_normal_alarm_count})
    # Validation model selection is exploratory and needs a new untouched test set.
    best=max(range(len(results)),key=lambda i:(results[i]['validation']['recall'],results[i]['validation']['precision'] or 0,
                                              -results[i]['validation']['false_positive_rate'],-i))
    return {'schema_version':'SensorExplorationResults/0.1','label_scope':scope,'config':config,'candidates':results,
        'selected_candidate_index':best,'selection_source':'validation_only','test_evaluated':False,
        'preprocessing_source':'train_only','versions':{'sklearn':sklearn.__version__,'numpy':np.__version__},'deployment_authorized':False,'physical_health_established':False,
        'episodes':{s:[e['episode_id'] for e in rows] for s,rows in splits.items()},
        'limits':'Grid search reuses validation and is optimistic. New held-out test and scope review required. Window FPR budget is not episode-level FPR or a physical safety limit. Warmup differs across window lengths; do not erase it from deployment accounting.'}


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def invalid(value):raise ValueError('nonfinite JSON')
    raw=args.episodes.read_bytes();configuration=args.config.read_bytes()
    result=explore(json.loads(raw,parse_constant=invalid),json.loads(configuration,parse_constant=invalid))
    result['episodes_input_sha256']=hashlib.sha256(raw).hexdigest()
    result['config_input_sha256']=hashlib.sha256(configuration).hexdigest()
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
