"""D v2 sensor-fault model. D-only seeded augmentation stays in training lineage."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from .b05_model import B05Model
class B05Supervised(B05Model):
    def _features(self,window):
        y=self._numeric(window)
        if y is None:return None
        return np.r_[y[:,0],y.mean(0),y.std(0),y.min(0),y.max(0),y[-1]-y[0]]
    def _score(self,window):
        f=self._features(window)
        return None if f is None else float(self.forest.predict_proba(f[None])[0,1])
    def fit(self,train,calibration,threshold):
        roles={}
        for role,es in [('train',train),('calibration',calibration),('threshold',threshold)]:
            if not es:raise ValueError('fit role empty')
            for e in es:
                if e['split']=='test' or (role=='train')!=(e['split']=='train'):raise ValueError('wrong fit split')
                if e['group_id'] in roles and roles[e['group_id']]!=role:raise ValueError('source lineage leaked')
                roles[e['group_id']]=role
        rng=np.random.default_rng(251126);xx=[];yy=[]
        # Training values ONLY: mirror development-declared injection family.
        index=self.indices[-1];sigmas=(.25,.4) if self.environment=='AtropineEnvGym' else (.03,.05)
        for e in train:
            for w,label in e['windows']:
                if label:raise ValueError('source training must be clean')
                x=np.asarray(w,dtype=float)
                if self._numeric(x) is None:continue
                xx.append(self._features(x));yy.append(False)
                for kind in ('bias','noise'):
                    for _ in range(8):
                        a=x.copy();length=int(rng.integers(1,5));start=int(rng.integers(0,5-length))
                        if kind=='bias':a[start:start+length,index]*=1.2
                        else:a[start:start+length,index]+=rng.normal(0,float(rng.choice(sigmas)),length)
                        xx.append(self._features(a));yy.append(True)
        if self.environment=='MAbEnvGym':
            # Additional NORMAL trajectories from the published PI + exact buffer
            # balance over the predeclared +/-0.3dm range. Derived model data,
            # not independent simulator samples or physical-health labels.
            for offset in np.linspace(-.3,.3,81):
                h=15.+offset;integral=0.;levels=[]
                for _ in range(10):
                    error=h-15.;integral+=error*60.
                    h-=60./(np.pi*2.5**2)*(.2777777777777778*error+4.62962962962963e-05*integral)
                    levels.append(h/3.)
                for start in range(7):
                    w=np.zeros((4,1970));w[:,17]=levels[start:start+4]
                    xx.append(self._features(w));yy.append(False)
                    for kind in ('bias','noise'):
                        for _ in range(2):
                            a=w.copy();length=int(rng.integers(1,5));at=int(rng.integers(0,5-length))
                            if kind=='bias':a[at:at+length,17]*=1.2
                            else:a[at:at+length,17]+=rng.normal(0,float(rng.choice(sigmas)),length)
                            xx.append(self._features(a));yy.append(True)
        self.forest=RandomForestClassifier(n_estimators=128,min_samples_leaf=2,class_weight='balanced',random_state=251126,n_jobs=1).fit(xx,yy)
        def rows(es):
            ss=[];ys=[]
            for e in es:
                for w,y in e['windows']:
                    s=self._score(w)
                    if s is not None:ss.append(s);ys.append(y)
            if not any(ys) or all(ys):raise ValueError('two classes required')
            return np.array(ss),np.array(ys)
        cs,cy=rows(calibration);self.calibrator=IsotonicRegression(out_of_bounds='clip').fit(cs,cy.astype(float))
        ts,ty=rows(threshold);self.threshold=max(.5,float(np.nextafter(ts[~ty].max(),np.inf)))
        self.fit_roles={k:[e['episode_id'] for e in v] for k,v in [('train',train),('calibration',calibration),('threshold',threshold)]}
        return self
