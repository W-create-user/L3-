"""Sensor-bias research model: Isolation Forest plus reference-relative residual.

No physical health probability or process safety threshold is inferred here.
Training, probability calibration and decision-threshold subsets are disjoint
by source episode; fit rejects test input entirely.
"""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from .anomaly_detector import confusion
from .event_metrics import evaluate_events, aggregate_events


class ResidualDetector:
    window_size=4
    dimension=39
    label_scope='sensor_fault'

    def _features(self, observations):
        x=np.asarray(observations,dtype=float)
        if x.ndim!=2 or x.shape[1]!=39 or len(x)<4 or not np.isfinite(x).all() or np.any(x[:,5]<=0):
            raise ValueError('four finite raw39 observations and positive reference required')
        ratio=x[:,6]/x[:,5]
        windows=np.array([ratio[i-3:i+1] for i in range(3,len(ratio))])
        residual=np.max(np.abs(windows-self.center),axis=1)/self.scale
        features=np.column_stack((windows.mean(1),windows.std(1),windows.min(1),windows.max(1),
                                  windows[:,-1]-windows[:,0]))
        return features,residual

    def _scores(self, observations):
        features,residual=self._features(observations)
        forest=-self.forest.score_samples(self.scaler.transform(features))
        return forest*residual,forest,residual

    def fit(self, episodes, *, binding):
        if any(e.get('split')=='test' for e in episodes):raise ValueError('test data forbidden during fit')
        parts={'train':[],'calibration':[],'threshold':[]};groups={};ids=set();signatures={}
        for e in episodes:
            part=e.get('fit_role');group=e['lineage_group'];split=e['split']
            if (part not in parts or e.get('label_scope')!='sensor_fault' or e['episode_id'] in ids
                    or split not in ('train','validation') or (part=='train')!=(split=='train')):
                raise ValueError('explicit train/calibration/threshold episode roles required')
            if group in groups and groups[group]!=part:raise ValueError('source lineage crosses fit subsets')
            x=np.asarray(e['observations'],dtype=float);y=e['labels']
            if (x.ndim!=2 or x.shape[1]!=39 or len(x)<4 or not np.isfinite(x).all()
                    or len(y)!=len(x) or any(type(v)is not bool for v in y) or np.any(x[:,5]<=0)):
                raise ValueError('finite source data and explicit labels required')
            signature=hashlib.sha256(x[:,1:].tobytes()).hexdigest()
            if signature in signatures and signatures[signature]!=part:raise ValueError('identical data crosses fit subsets')
            if part=='train' and any(y):raise ValueError('train must be uncorrupted')
            groups[group]=part;signatures[signature]=part;ids.add(e['episode_id']);parts[part].append(e)
        if not all(parts.values()):raise ValueError('three nonempty fit subsets required')
        clean=np.concatenate([np.asarray(e['observations'],dtype=float) for e in parts['train']])
        ratio=clean[:,6]/clean[:,5];self.center=float(np.median(ratio))
        self.scale=max(float(np.max(np.abs(ratio-self.center))),1e-8)
        features=np.concatenate([self._features(e['observations'])[0] for e in parts['train']])
        self.scaler=StandardScaler().fit(features)
        self.forest=IsolationForest(n_estimators=100,contamination='auto',random_state=1201,n_jobs=1)
        self.forest.fit(self.scaler.transform(features))
        def rows(subset):
            scores=[];labels=[]
            for e in subset:
                scores.extend(self._scores(e['observations'])[0]);y=e['labels']
                labels.extend(any(y[i-3:i+1]) for i in range(3,len(y)))
            labels=np.asarray(labels,dtype=bool)
            if not labels.any() or labels.all():raise ValueError('both labels required in validation subsets')
            return np.asarray(scores),labels
        cs,cy=rows(parts['calibration'])
        self.calibrator=IsotonicRegression(out_of_bounds='clip').fit(cs,cy.astype(float))
        ts,ty=rows(parts['threshold'])
        # Predeclared zero false alarms on validation-negative windows, then
        # highest recall. This is a selection rule, not a population guarantee.
        candidates=np.unique(np.r_[ts,np.nextafter(ts.max(),np.inf)])
        feasible=[]
        for t in candidates:
            c=confusion(ty,ts>=t)
            if c['FP']==0:feasible.append((c['recall'],float(t),c))
        recall,self.threshold,counts=max(feasible,key=lambda a:(a[0],a[1]))
        self.binding=binding
        ref=clean[:,5];margin=.1*max(float(np.ptp(ref)),float(abs(np.median(ref))),1e-8)
        self.reference_domain=(float(ref.min()-margin),float(ref.max()+margin))
        self.card={'schema_version':'DResidualDetector/1.0','window_size':4,'seed':1201,'trees':100,
            'center':self.center,'scale':self.scale,'threshold':self.threshold,
            'feature_indices':[5,6],'excluded':'clock, labels, split, seeds, raw audit channels and future observations',
            'method':'IsolationForest anomaly score multiplied by maximum reference-ratio residual',
            'probability_meaning':'isotonic estimate of artificial sensor-bias window label; not physical safety or health',
            'fit_episode_ids':{k:[e['episode_id'] for e in v] for k,v in parts.items()},
            'threshold_policy':'zero FP on threshold-validation negative windows, maximize recall, higher threshold tie break',
            'threshold_validation':counts,'binding':binding,'reference_domain':self.reference_domain,
            'domain_basis':'training reference range with 10% empirical margin; applicability heuristic, not physical bound',
            'deployment_authorized':False,'physical_safety_certified':False}
        return self.card

    def score(self, window):
        x=np.asarray(window,dtype=float)
        if x.shape!=(4,39):raise ValueError('exact 4x39 window required')
        scores,forest,residual=self._scores(x)
        return {'anomaly_score':float(scores[0]),'forest_score':float(forest[0]),
                'relative_residual':float(residual[0]),'threshold':self.threshold,
                'anomalous':bool(scores[0]>=self.threshold),
                'sensor_bias_probability':float(self.calibrator.predict(scores)[0]),
                'physical_safety_probability':None,
                'in_empirical_reference_domain':bool(self.reference_domain[0]<=x[-1,5]<=self.reference_domain[1])}

    def evaluate(self, episodes):
        reports=[];all_y=[];all_p=[]
        for e in episodes:
            scores,forest,residual=self._scores(e['observations']);pred=scores>=self.threshold
            truth=np.array([any(e['labels'][i-3:i+1]) for i in range(3,len(e['labels']))])
            probability=self.calibrator.predict(scores)
            all_y.extend(truth);all_p.extend(probability)
            reports.append({'episode_id':e['episode_id'],'counts':confusion(truth,pred),
                'event':evaluate_events(e['labels'],pred,4),
                'whole_normal':not any(e['labels']),'any_alarm':bool(pred.any())})
        y=np.array(all_y);p=np.array(all_p);bins=[]
        for i in range(10):
            mask=(p>=i/10)&(p<(i+1)/10) if i<9 else (p>=.9)&(p<=1)
            if mask.any():bins.append({'count':int(mask.sum()),'mean_probability':float(p[mask].mean()),'label_rate':float(y[mask].mean())})
        normal=[r for r in reports if r['whole_normal']]
        return {'episodes':reports,'counts':{k:sum(r['counts'][k] for r in reports) for k in ('TP','FP','FN','TN')},
                'events':aggregate_events([r['event'] for r in reports]),
                'whole_normal_episodes':len(normal),'whole_normal_with_alarm':sum(r['any_alarm'] for r in normal),
                'brier_score':float(np.mean((p-y)**2)),
                'ece':sum(b['count']/len(y)*abs(b['mean_probability']-b['label_rate']) for b in bins),
                'calibration_bins':bins,'confidence_interpretation':'correlated window diagnostic; not independent probability validation'}

    def save(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:pickle.dump(self,stream,protocol=5)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load_trusted(cls,path,expected_sha256):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('trusted model hash mismatch')
        model=pickle.loads(path.read_bytes())
        if not isinstance(model,cls):raise ValueError('unexpected trusted model type')
        return model
