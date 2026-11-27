"""D research-only B05 models. Missing windows abstain; no safety certification."""
import base64
import hashlib
import json
import pickle
import time
import zlib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from .binding import fingerprint

ENV={'AtropineEnvGym':(39,[5,6]),'MAbEnvGym':(1970,[17])}

def check_packet(packet, context, binding):
    if not isinstance(packet,dict) or not isinstance(context,dict):raise ValueError('packet/context required')
    if packet.get('binding')!=binding or context.get('binding')!=binding:raise ValueError('binding mismatch')
    eid=context.get('episode_id');v=context.get('state_version')
    if not isinstance(eid,str) or not eid or type(v)is not int or v<3:raise ValueError('live episode/version required')
    if (packet.get('episode_id')!=eid or not isinstance(packet.get('state_versions'),list) or any(type(n)is not int for n in packet['state_versions']) or packet['state_versions']!=list(range(v-3,v+1))):raise ValueError('stale/noncontiguous window')
    observed=context.get('observed_monotonic')
    if type(observed) not in (int,float) or not np.isfinite(observed) or not 0<=time.monotonic()-observed<=.5:raise ValueError('stale observation age')
    return packet['values']

class B05Model:
    def __init__(self,environment,binding):
        if environment not in ENV:raise ValueError('unknown environment')
        self.environment=environment;self.dimension,self.indices=ENV[environment];self.binding=binding
    def _numeric(self,window):
        raw=np.asarray(window,dtype=object)
        if raw.shape!=(4,self.dimension):raise ValueError('wrong full observation shape')
        # All coordinates must be present, including those outside model projection.
        if any(v is None for v in raw.flat):return None
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in raw.flat):raise ValueError('numeric nonboolean observation required')
        x=np.asarray(raw,dtype=float)
        if not np.isfinite(x).all():return None
        if self.environment=='AtropineEnvGym':
            if np.any(x[:,5]<=0):raise ValueError('nonpositive reference')
            y=(x[:,6]/x[:,5])[:,None]
        else:y=x[:,[17]]*3.  # explicit learned scalar physical projection, NOT a 39-D model reuse
        return y
    def _features(self,window):
        y=self._numeric(window)
        if y is None:return None
        return np.r_[y.mean(0),y.std(0),y.min(0),y.max(0),y[-1]-y[0]]
    def _score(self,window):
        y=self._numeric(window);features=self._features(window)
        if y is None:return None
        z=(features-self.feature_mean)/self.feature_std
        residual=float(np.max(np.abs(y-self.center)/self.scale))
        return float(-self.forest.score_samples(z[None])[0]*residual)
    def fit(self,train,calibration,threshold):
        groups={}
        for role,episodes in [('train',train),('calibration',calibration),('threshold',threshold)]:
            if not episodes:raise ValueError('nonempty disjoint fit roles required')
            for e in episodes:
                if e['split']=='test' or (role=='train')!=(e['split']=='train'):raise ValueError('test or wrong split during fit')
                g=e['group_id']
                if g in groups and groups[g]!=role:raise ValueError('cross-role source lineage')
                groups[g]=role
        windows=[w for e in train for w,label in e['windows'] if self._numeric(w) is not None]
        if any(label for e in train for _,label in e['windows']):raise ValueError('clean training required')
        values=np.concatenate([self._numeric(w) for w in windows]);self.center=np.median(values,axis=0)
        self.scale=np.maximum(np.max(np.abs(values-self.center),axis=0),1e-6)
        features=np.stack([self._features(w) for w in windows]);self.feature_mean=features.mean(0);self.feature_std=features.std(0).clip(1e-6)
        self.forest=IsolationForest(n_estimators=64,random_state=251125,n_jobs=1).fit((features-self.feature_mean)/self.feature_std)
        def rows(episodes):
            ss=[];yy=[]
            for e in episodes:
                for w,label in e['windows']:
                    s=self._score(w)
                    if s is not None:ss.append(s);yy.append(label)
            if not any(yy) or all(yy):raise ValueError('both finite label classes required')
            return np.asarray(ss),np.asarray(yy)
        cs,cy=rows(calibration);self.calibrator=IsotonicRegression(out_of_bounds='clip').fit(cs,cy.astype(float))
        ts,ty=rows(threshold)
        # Fixed policy: lowest finite boundary above every threshold-validation negative.
        self.threshold=float(np.nextafter(ts[~ty].max(),np.inf))
        self.fit_roles={k:[e['episode_id'] for e in v] for k,v in [('train',train),('calibration',calibration),('threshold',threshold)]}
        return self
    def score_window(self,window):
        score=self._score(window)
        if score is None:return {'status':'unavailable','reason':'missing_or_nonfinite_window','anomalous':None,
            'automatic_execution_blocked':True,'execution_authorized':False,'physical_safety_probability':None}
        return {'status':'scored','anomaly_score':score,'threshold':self.threshold,'anomalous':bool(score>=self.threshold),
            'synthetic_fault_probability':float(self.calibrator.predict([score])[0]),
            'automatic_execution_blocked':bool(score>=self.threshold),'execution_authorized':False,
            'physical_safety_probability':None}
    def score_packet(self,packet,context):
        try:return self.score_window(check_packet(packet,context,self.binding))
        except (ValueError,TypeError,KeyError):
            return {'status':'unavailable','reason':'invalid_or_stale_bound_window','anomalous':None,
                    'automatic_execution_blocked':True,'execution_authorized':False,'physical_safety_probability':None}
    def save(self,path):
        # Text envelope for portable receipt. Trust must come from external receipt SHA.
        payload=pickle.dumps(self,protocol=5)
        card={'schema_version':'DB05Model/1.0','environment':self.environment,'binding':self.binding,
            'label_scope':'synthetic_sensor_fault','window_label':'any source fault in four observations',
            'window_size':4,'fit_roles':self.fit_roles,'threshold':self.threshold,'payload_sha256':hashlib.sha256(payload).hexdigest(),
            'deployment_authorized':False,'physical_safety_certified':False,
            'payload_zlib_base64':base64.b64encode(zlib.compress(payload)).decode()}
        with open(path,'x') as f:json.dump(card,f,sort_keys=True,allow_nan=False);f.write('\n')
        with open(path,'rb') as stream:return hashlib.sha256(stream.read()).hexdigest()
    @classmethod
    def load_trusted(cls,path,expected_sha256):
        with open(path,'rb') as stream:raw=stream.read()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('untrusted model envelope')
        card=json.loads(raw);payload=zlib.decompress(base64.b64decode(card['payload_zlib_base64']))
        if hashlib.sha256(payload).hexdigest()!=card['payload_sha256']:raise ValueError('model payload changed')
        model=pickle.loads(payload)
        if not isinstance(model,cls) or model.binding!=card['binding']:raise ValueError('wrong model binding')
        return model
