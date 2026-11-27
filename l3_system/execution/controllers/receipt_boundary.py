"""B-owned opt-in boundary around the frozen 68831ef controller.

The sole environment owner MUST route predict, review binding, step, receipt
and commit through this object. Malformed/missing/stale receipts latch both
controller proposals and this object's native-step entry. This is an audited
same-process boundary, not an OS sandbox against arbitrary Python access.
"""
import copy
from datetime import datetime
import math
import time
import numpy as np


class ReceiptBoundary:
    RECEIPT_FIELDS=frozenset(('request_id','episode_id','step_count','state_version','success','action'))

    def __init__(self, adapter, policy, certificate, *, receipt_timeout_seconds, clock=time.monotonic):
        self.adapter=adapter;self.policy=policy;self.clock=clock
        self.failed=False;self.request=None;self.expected=None;self.received=False;self.step_started=False
        if (type(receipt_timeout_seconds) not in (int,float) or not math.isfinite(receipt_timeout_seconds)
                or receipt_timeout_seconds<=0):raise ValueError('Explicit finite positive receipt deadline required')
        self.timeout=float(receipt_timeout_seconds)
        required={'episode_id','state_version','timestamp','level_dm','calibration_source'}
        if not isinstance(certificate,dict) or set(certificate)!=required:raise ValueError('Complete startup certificate required')
        cert=copy.deepcopy(certificate)
        if (type(cert['state_version'])is not int or cert['episode_id']!=adapter.metadata['episode_id']
                or cert['state_version']!=adapter.state_version or adapter.step_count!=0
                or cert['calibration_source']!='validated_simulator_reset_before_fault_injection'
                or type(cert['level_dm']) not in (int,float) or not math.isfinite(cert['level_dm'])
                or cert['level_dm']!=policy.anchor[0] or (cert['episode_id'],cert['state_version'])!=policy.anchor[1:]):
            raise ValueError('Certificate does not match fresh trusted-reset anchor')
        stamp=datetime.fromisoformat(cert['timestamp'])
        if stamp.tzinfo is None:raise ValueError('Startup timestamp must include timezone')
        self.certificate=cert
        self.audit=[]

    def __getattr__(self,name):return getattr(self.adapter,name)

    @staticmethod
    def _action(value):
        action=np.asarray(value)
        if action.shape!=(9,) or action.dtype.kind not in 'fiu' or not np.isfinite(action).all():
            raise ValueError('Finite numeric nine-action array required')
        return action.astype(float)

    def _stop(self,reason):
        self.failed=True;self.policy.failed=True
        self.audit.append({'event':'latched','reason':reason,'step_calls':self.adapter.step_calls})
        raise RuntimeError(reason)

    def _active(self):
        if self.failed:self._stop('B boundary latched; explicit new reset certificate and instance required')

    def reset(self):
        self._active()
        if self.request is not None or self.step_started:self._stop('Cannot reset an active receipt transaction')
        self.policy.reset()

    def predict(self,observation,space):
        self._active()
        if self.step_started or self.policy.pending is not None:self._stop('Previous execution receipt or commit missing')
        try:return self.policy.predict(observation,space)
        except Exception as exc:self._stop('Candidate proposal failed: '+type(exc).__name__)

    def bind_request(self,request):
        """Call from BOTH D review callbacks; identical repeated binding is harmless."""
        self._active()
        try:
            if not isinstance(request,dict) or self.policy.pending is None:raise ValueError('No candidate/request')
            keys=('request_id','episode_id','state_version','step_count')
            compact={key:request[key] for key in keys}
            if (not isinstance(compact['request_id'],str) or not compact['request_id']
                    or compact['episode_id']!=self.adapter.metadata['episode_id']
                    or type(compact['state_version']) is not int or compact['state_version']!=self.adapter.state_version
                    or type(compact['step_count'])is not int or compact['step_count']!=self.adapter.step_count
                    or not np.array_equal(self._action(request['action']),self.policy.pending)):
                raise ValueError('Request identity/action mismatch')
            if self.request is not None and self.request!=compact:raise ValueError('Request changed after binding')
            self.request=compact
        except Exception as exc:self._stop('Invalid request binding: '+type(exc).__name__)

    def step(self,action):
        self._active()
        if self.step_started or self.request is None or self.policy.pending is None:self._stop('Step needs a fresh reviewed request; prior receipt missing')
        try:
            if (self.request['episode_id']!=self.adapter.metadata['episode_id']
                    or self.request['state_version']!=self.adapter.state_version
                    or self.request['step_count']!=self.adapter.step_count
                    or not np.array_equal(self._action(action),self.policy.pending)):
                raise ValueError('Stale or changed native-step request')
        except Exception as exc:self._stop('Step context mismatch: '+type(exc).__name__)
        self.step_started=True;self.received=False
        try:
            result=self.adapter.step(action)
            # Retain only a private expected receipt; scoring truth remains with caller.
            self.expected={**self.request,'step_count':self.request['step_count']+1,
                'state_version':self.request['state_version']+1,'success':True,
                'action':self.policy.pending.tolist()}
            self.deadline=self.clock()+self.timeout
            if (result.get('success') is not True or result.get('step_count')!=self.expected['step_count']
                    or result.get('state_version')!=self.expected['state_version']
                    or not np.array_equal(result.get('action'),self.policy.pending)):
                raise ValueError('Native step returned unknown or invalid execution')
            return result
        except Exception as exc:self._stop('Unknown native execution: '+type(exc).__name__)

    def make_receipt(self,result):
        """Owner-side allowlist projection: no observation, reward, labels or schedule."""
        self._active()
        if self.expected is None:self._stop('No native result awaiting receipt')
        try:
            return {'request_id':self.request['request_id'],'episode_id':self.request['episode_id'],
                **{key:copy.deepcopy(result[key]) for key in ('step_count','state_version','success','action')}}
        except Exception as exc:self._stop('Missing native receipt fields: '+type(exc).__name__)

    def acknowledge(self,receipt):
        self._active()
        try:
            if self.expected is None or self.received or self.clock()>self.deadline:raise ValueError('Missing, duplicate or expired receipt')
            if not isinstance(receipt,dict) or set(receipt)!=self.RECEIPT_FIELDS:raise ValueError('Allowlisted receipt fields required')
            if type(receipt['success'])is not bool or receipt['success'] is not True:raise ValueError('Successful execution required')
            for key in ('state_version','step_count'):
                if type(receipt[key])is not int:raise ValueError('Strict integer identity required')
            if any(receipt[key]!=self.expected[key] for key in self.RECEIPT_FIELDS-{'action'}):raise ValueError('Receipt identity mismatch')
            action=self._action(receipt['action'])
            if action.shape!=(9,) or not np.isfinite(action).all() or not np.array_equal(action,self.policy.pending):raise ValueError('Receipt action mismatch')
            if (self.adapter.metadata['episode_id']!=receipt['episode_id']
                    or self.adapter.state_version!=receipt['state_version'] or self.adapter.step_count!=receipt['step_count']):
                raise ValueError('Receipt/live epoch mismatch')
            self.policy.acknowledge({'success':True,'action':action.tolist(),'state_version':receipt['state_version']})
            self.received=True
            self.audit.append({'event':'receipt_accepted','request_id':receipt['request_id'],
                               'state_version':receipt['state_version'],'step_count':receipt['step_count']})
        except Exception as exc:self._stop('Invalid execution receipt: '+type(exc).__name__)

    def commit(self,executed):
        self._active()
        if executed is True and not self.received:self._stop('Missing acknowledgement before commit')
        if executed is False and self.step_started:self._stop('Physical step cannot be treated as unexecuted')
        try:self.policy.commit(executed)
        except Exception as exc:self._stop('Candidate commit failed: '+type(exc).__name__)
        self.request=None;self.expected=None;self.received=False;self.step_started=False
