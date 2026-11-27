"""D26 opt-in DEVELOPMENT bridge: hard rules + bound learned gate + recovery.

Never steps the backend. B05->B26 extension is an explicit experimental scope,
not a claim of controller-independent model validation or formal approval.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from .b05_model import B05Model
from .inference_service import InferenceService
from .risk_policy import RiskEvidence,RiskPolicy

def validate_anchor(certificate,adapter):
    if not isinstance(certificate,dict):raise ValueError('startup certificate required')
    if (certificate.get('episode_id')!=adapter.metadata['episode_id'] or certificate.get('state_version')!=adapter.state_version
        or certificate.get('unit')!='dm' or certificate.get('coordinate')!='physical_projection'
        or certificate.get('kind')!='trusted_simulation_reset_before_fault' or not certificate.get('source_ref')
        or certificate.get('fault_started') is not False):raise ValueError('untrusted or post-fault anchor')
    t=certificate.get('observed_monotonic');h=certificate.get('level_dm')
    if type(t) not in (int,float) or not 0<=time.monotonic()-t<=5 or type(h) not in (int,float) or not np.isfinite(h):raise ValueError('stale/invalid startup anchor')
    if not np.isclose(h,float(adapter.get_observation()[17]*adapter._xscale[17]),rtol=0,atol=1e-6):raise ValueError('startup reference mismatch')
    return h

class FailClosedPolicy:
    """D-owned outer transaction latch; never changes B controller internals."""
    def __init__(self,base):self.base=base;self.failed=False
    def __getattr__(self,name):return getattr(self.base,name)
    def _call(self,name,*args):
        if self.failed:raise RuntimeError('D receipt guard latched; no retry')
        try:return getattr(self.base,name)(*args)
        except Exception:
            self.failed=True;raise
    def reset(self):return self._call('reset')
    def predict(self,*args):return self._call('predict',*args)
    def acknowledge(self,result):return self._call('acknowledge',result)
    def commit(self,executed):return self._call('commit',executed)

class B26LearningGate:
    def __init__(self,adapter,hard_monitor,model_path,model_sha256,config_path,config_sha256,*,source_root,development_opt_in=False,resources_clear=lambda:False):
        if development_opt_in is not True:raise ValueError('joint approval absent; only explicit development opt-in supported')
        raw=Path(config_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config_sha256:raise ValueError('gate config hash mismatch')
        self.config=json.loads(raw)
        if self.config['model_sha256']!=model_sha256 or self.config['formal_launch_authorized'] is not False:raise ValueError('invalid development release scope')
        for name,want in self.config['runtime_sha256'].items():
            if hashlib.sha256((Path(source_root)/name).read_bytes()).hexdigest()!=want:raise ValueError('runtime source changed: '+name)
        if hard_monitor.process_rules is None or not hard_monitor.process_rules.enabled:raise ValueError('explicit hard model fence required')
        self.model=B05Model.load_trusted(model_path,model_sha256);self.backend=adapter;self.hard=hard_monitor;self.clear=resources_clear
        self.service=InferenceService(model_path,model_sha256,timeout_seconds=self.config['inference_seconds'])
        self.policy=RiskPolicy();self.episode=adapter.metadata['episode_id'];self.policy.reset(self.episode)
        self.values=[];self.versions=[];self.observed=None;self.failed=False;self.last=None;self.events=[];self.cache={}
    def __getattr__(self,name):return getattr(self.hard,name)
    def _event(self,kind,**fields):
        self.events.append({'kind':kind,'episode_id':self.episode,'state_version':self.backend.state_version,'monotonic':time.monotonic(),**fields})
    def _fault(self,reason):
        self.failed=True;self.last=self.policy._latch(reason);self._event('takeover',reason=reason)
    def _current(self):
        return (not self.failed and not self.backend.done and self.backend.metadata['episode_id']==self.episode and self.versions
            and self.versions[-1]==self.backend.state_version and self.observed is not None
            and 0<=time.monotonic()-self.observed<=self.config['observation_max_age_seconds']
            and np.array_equal(self.backend.get_observation(),self.values[-1]))
    def _score(self,ident):
        packet={'binding':self.model.binding,'episode_id':self.episode,'state_versions':self.versions,'values':self.values}
        context={'binding':self.model.binding,'episode_id':self.episode,'state_version':self.backend.state_version,'observed_monotonic':self.observed}
        return self.service.score(ident,packet,context)
    def observe_executed(self,result):
        obs=np.asarray(self.backend.get_observation())
        if (result.get('success') is not True or result.get('state_version')!=self.backend.state_version
            or self.backend.metadata['episode_id']!=self.episode or obs.shape!=(1970,) or not np.isfinite(obs).all()
            or not np.array_equal(obs,result.get('observation'))
            or self.versions and self.backend.state_version!=self.versions[-1]+1):
            self._fault('invalid_or_unknown_executed_state');raise ValueError('fresh successful step required')
        self.values=(self.values+[obs.tolist()])[-4:];self.versions=(self.versions+[self.backend.state_version])[-4:];self.observed=time.monotonic()
        if len(self.values)<4:self._event('restricted_history',records=len(self.values));return
        score=self._score('post-'+str(self.backend.state_version))
        if score['status']!='scored':self._fault(score.get('reason','model_unavailable'));return
        hard=self.hard.check_observation(obs)['approved']
        self.last=self.policy.evaluate(RiskEvidence(self.episode,self.backend.state_version,hard,'valid',True,score['anomalous'],'in_scope'))
        self._event('post_review',model=score,risk=self.last)
        if self.last['risk_level']==3:self.failed=True
    def confirm(self,operator_ref):
        if not self._current() or not self.clear():raise ValueError('current verified state and released resources required')
        self.last=self.policy.confirm_recovery(self.episode,state_version=self.backend.state_version,reviewer_ref=operator_ref)
        self._event('human_confirmation',operator_ref=operator_ref,step_delta=0)
    def check_request(self,request,observation=None):
        hard=self.hard.check_request(request,observation)
        def reject(reason,level=3):
            return {**hard,'decision':'reject','rule_id':'D26-LEARNED-GATE','reason':reason,'risk_level':level}
        if hard['decision']!='pass':self._fault('hard_rule_rejection');return hard
        if len(self.values)<4:return reject('restricted_fresh_history_required',2)
        if not self._current():self._fault('stale_or_unknown_live_context');return reject('stale_or_unknown_live_context')
        if request.get('episode_id')!=self.episode or request.get('state_version')!=self.backend.state_version:return reject('request_context_mismatch')
        if self.last is None or self.last['risk_level']!=1:return reject('manual_recovery_review_required',2)
        ident=request.get('request_id')
        signature=json.dumps({k:request.get(k) for k in ('request_id','task_id','subtask_id','episode_id','state_version','action')},sort_keys=True,allow_nan=False)
        if ident in self.cache:
            old_signature,old_decision=self.cache[ident]
            if signature!=old_signature:self._fault('changed_reused_request');return reject('changed_reused_request')
            self._event('idempotent_review',request_id=ident)
            return copy.deepcopy(old_decision)
        score=self._score('pre-'+str(ident))
        self._event('pre_review',request_id=request.get('request_id'),model=score)
        if score['status']!='scored' or score['anomalous']:
            self._fault(score.get('reason','statistical_anomaly'));return reject('learned_model_blocked')
        decision={**hard,'model_sha256':self.config['model_sha256'],'gate_scope':'B26_development_extension_only'}
        self.cache[ident]=(signature,copy.deepcopy(decision));return decision
    def close(self):self.service.close()
