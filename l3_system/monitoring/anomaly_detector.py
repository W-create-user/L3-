"""Normal-only Isolation Forest with grouped episodes and validation-only threshold.

Scores are anomaly scores, never probabilities or certified physical limits.
"""
import json
from pathlib import Path
import pickle
import hashlib
from copy import deepcopy
from .binding import fingerprint

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def features(window):
    values=np.asarray(window)
    if values.dtype.kind not in 'fiu' or values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('finite numeric time-by-variable window of length >=2 required')
    return np.concatenate([values.mean(0),values.std(0),values.min(0),values.max(0),
                           values[-1]-values[0],np.abs(np.diff(values,axis=0)).mean(0)])


def confusion(labels,predictions):
    labels=np.asarray(labels,dtype=bool);predictions=np.asarray(predictions,dtype=bool)
    tp=int(np.sum(labels&predictions));fp=int(np.sum(~labels&predictions))
    fn=int(np.sum(labels&~predictions));tn=int(np.sum(~labels&~predictions))
    ratio=lambda n,d:n/d if d else None
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                false_positive_rate=ratio(fp,fp+tn))


class AnomalyDetector:
    def __init__(self,window_size=4,seed=1201):
        if type(window_size) is not int or window_size<2:raise ValueError('window_size >=2 required')
        self.window_size=window_size;self.seed=seed;self.threshold=None

    def _windows(self,episode):
        values=np.asarray(episode['observations'])
        if values.ndim!=2 or len(values)<self.window_size or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('episode has invalid/insufficient observations')
        if hasattr(self,'dimension') and values.shape[1]!=self.dimension:raise ValueError('variable dimension mismatch')
        labels=episode['labels']
        if len(labels)!=len(values) or any(type(x) is not bool for x in labels):raise ValueError('one explicit boolean label per observation required')
        windows=[features(values[i-self.window_size+1:i+1]) for i in range(self.window_size-1,len(values))]
        # A window is anomalous if ANY labelled sample in it is anomalous.
        truth=[any(labels[i-self.window_size+1:i+1]) for i in range(self.window_size-1,len(values))]
        return np.asarray(windows),np.asarray(truth,dtype=bool)

    def fit(self,episodes,*,label_scope,binding=None,receipt=None):
        if label_scope not in ('synthetic_fixture','sensor_fault','process_anomaly'):raise ValueError('explicit label scope required')
        groups={};ids=set();hashes={};splits={'train':[],'validation':[],'test':[]}
        for episode in episodes:
            eid=episode['episode_id'];group=episode['lineage_group'];split=episode['split']
            if not isinstance(eid,str) or not eid.strip() or eid in ids:raise ValueError('unique episode_id required')
            if not isinstance(group,str) or not group.strip() or split not in splits:raise ValueError('source group and split required')
            if not isinstance(episode.get('label_source'),str) or not episode['label_source'].strip():raise ValueError('independent label source declaration required')
            if episode.get('label_scope')!=label_scope:raise ValueError('do not mix sensor/process/fixture label scopes')
            if group in groups and groups[group]!=split:raise ValueError('lineage crosses splits')
            groups[group]=split;ids.add(eid)
            signature=hashlib.sha256(json.dumps(np.asarray(episode['observations'],dtype=float).tolist(),allow_nan=False).encode()).hexdigest()
            if signature in hashes and hashes[signature]!=split:raise ValueError('identical trajectory crosses splits')
            hashes[signature]=split;splits[split].append(episode)
        if any(not value for value in splits.values()):raise ValueError('three nonempty episode splits required')
        self.dimension=np.asarray(splits['train'][0]['observations']).shape[-1]
        data={split:[(episode,*self._windows(episode)) for episode in subset] for split,subset in splits.items()}
        if any(np.any(labels) for _,_,labels in data['train']):raise ValueError('train must contain normal-only episodes')
        train=np.concatenate([x for _,x,_ in data['train']]);vx=np.concatenate([x for _,x,_ in data['validation']])
        vy=np.concatenate([y for _,_,y in data['validation']])
        if not vy.any() or vy.all():raise ValueError('validation requires both normal and anomalous windows')
        self.scaler=StandardScaler().fit(train)
        self.model=IsolationForest(n_estimators=100,contamination='auto',random_state=self.seed,n_jobs=1)
        self.model.fit(self.scaler.transform(train))
        scores=-self.model.score_samples(self.scaler.transform(vx))
        # Threshold candidates use validation ONLY; ties choose the higher threshold.
        candidates=np.unique(np.r_[scores,np.nextafter(scores.max(),np.inf)])
        def quality(threshold):
            m=confusion(vy,scores>=threshold);den=2*m['TP']+m['FP']+m['FN']
            return (2*m['TP']/den if den else 0,float(threshold))
        self.threshold=float(max(candidates,key=quality))
        self.label_scope=label_scope
        if binding is not None and len(binding['observation'])!=self.dimension:raise ValueError('binding dimension mismatch')
        self.binding=deepcopy(binding)
        reports={}
        for split in ('validation','test'):
            episode_reports=[]
            for episode,x,truth in data[split]:
                score=-self.model.score_samples(self.scaler.transform(x));pred=score>=self.threshold
                # Delay only for detected events; misses are separately retained.
                onset=np.flatnonzero(truth);delay=None
                if len(onset):
                    end=int(onset[0])
                    while end<len(truth) and truth[end]:end+=1
                    hits=np.flatnonzero(pred[int(onset[0]):end])+int(onset[0])
                    if len(hits):delay=int(hits[0]-onset[0])
                episode_reports.append(dict(episode_id=episode['episode_id'],windows=len(truth),
                    confusion=confusion(truth,pred),first_event_delay_records=delay,
                    first_event_missed=bool(truth.any() and delay is None),
                    anomalous_episode_missed=bool(truth.any() and not np.any(pred&truth))))
            reports[split]={'episodes':episode_reports,'counts':{key:sum(r['confusion'][key] for r in episode_reports)
                                                              for key in ('TP','FP','FN','TN')}}
        self.card={'label_scope':label_scope,'window_size':self.window_size,'dimension':self.dimension,
                   'binding':deepcopy(binding),'binding_sha256':fingerprint(binding) if binding is not None else None,
                   'intake_receipt':deepcopy(receipt),
                   'seed':self.seed,'threshold':self.threshold,'threshold_source':'validation_only',
                   'scaler_source':'train_only','versions':{'sklearn':sklearn.__version__,'numpy':np.__version__},
                   'episodes':{split:[e['episode_id'] for e in subset] for split,subset in splits.items()},
                   'evaluation':reports,'physical_safety_limit':False,'probability_calibrated':False,
                   'limits':'Window counts are correlated; first-event delay is in records, not physical seconds. Labels and lineage declarations need independent review.'}
        return self.card

    def score(self,window):
        if self.threshold is None:raise ValueError('model not trained')
        values=np.asarray(window)
        if values.shape!=(self.window_size,self.dimension):raise ValueError('window shape mismatch')
        x=features(values).reshape(1,-1)
        score=float(-self.model.score_samples(self.scaler.transform(x))[0])
        return {'anomaly_score':score,'threshold':self.threshold,'anomalous':score>=self.threshold,
                'label_scope':self.label_scope,'is_probability':False}

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        blob=pickle.dumps(self,protocol=5)
        (directory/'model.pkl').write_bytes(blob)
        (directory/'model_card.json').write_text(json.dumps(dict(self.card,model_sha256=hashlib.sha256(blob).hexdigest()),indent=2)+'\n')

    @classmethod
    def load_trusted(cls,directory,*,expected_binding=None):
        """Pickle is executable: load ONLY artifacts created in your trusted training run."""
        directory=Path(directory);card=json.loads((directory/'model_card.json').read_text())
        blob=(directory/'model.pkl').read_bytes()
        if hashlib.sha256(blob).hexdigest()!=card['model_sha256']:raise ValueError('model hash mismatch')
        if card['versions']['sklearn']!=sklearn.__version__:raise ValueError('sklearn version mismatch')
        if card['versions']['numpy']!=np.__version__:raise ValueError('numpy version mismatch')
        model=pickle.loads(blob)
        if not isinstance(model,cls):raise ValueError('unexpected model object')
        if expected_binding is not None and getattr(model,'binding',None)!=expected_binding:raise ValueError('loaded model semantic binding mismatch')
        return model
