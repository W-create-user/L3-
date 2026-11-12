"""D05 experimental risk advisory, not calibrated confidence or execution approval."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskEvidence:
    episode_id: str
    state_version: int
    hard_rule_approved: bool
    data_quality: str  # valid / degraded / unavailable
    context_current: bool
    anomaly: object  # bool, or None when scorer unavailable
    distribution: str  # in_scope / out_of_scope / unknown

    def validate(self):
        if (not isinstance(self.episode_id,str) or not self.episode_id.strip()
                or type(self.state_version)is not int or self.state_version<0
                or type(self.hard_rule_approved)is not bool or type(self.context_current)is not bool
                or self.data_quality not in ('valid','degraded','unavailable')
                or self.anomaly is not None and type(self.anomaly)is not bool
                or self.distribution not in ('in_scope','out_of_scope','unknown')):
            raise ValueError('typed, versioned risk evidence required')


class RiskPolicy:
    """Stateful advisory with explicit episode reset and manual recovery acknowledgement.

    All outputs still require the existing action gate. Level 2/3 block automatic
    execution. Level 1 only allows submitting a request for that gate to review.
    No probability, reward surrogate or anomaly score becomes a physical limit.
    """
    def __init__(self,clean_observations=3):
        if type(clean_observations)is not int or clean_observations<2:
            raise ValueError('recovery requires at least two independent state versions')
        self.required=clean_observations;self.episode=None;self.version=None
        self.clean=0;self.latched=True;self.ack=False

    def reset(self,episode_id):
        if not isinstance(episode_id,str) or not episode_id.strip():raise ValueError('episode ID required')
        self.episode=episode_id;self.version=None;self.clean=0;self.latched=True;self.ack=False

    def acknowledge_review(self,episode_id):
        if self.episode!=episode_id:raise ValueError('acknowledgement belongs to another episode')
        self.ack=True

    def evaluate(self,evidence):
        evidence.validate()
        if self.episode is None:self.reset(evidence.episode_id)
        if evidence.episode_id!=self.episode:raise ValueError('explicit reset required at episode boundary')
        if self.version is not None and evidence.state_version<=self.version:
            self.latched=True;self.clean=0;self.ack=False
            return self._result(3,'repeated_or_reordered_state')
        contiguous=self.version is None or evidence.state_version==self.version+1
        self.version=evidence.state_version
        if not evidence.hard_rule_approved:
            return self._latch('hard_rule_rejection')
        if not evidence.context_current or evidence.data_quality=='unavailable':
            return self._latch('unavailable_or_stale_evidence')
        if evidence.anomaly is None or evidence.distribution!='in_scope':
            return self._latch('model_scope_or_verdict_unknown')
        if evidence.anomaly:
            return self._latch('statistical_anomaly_requires_review')
        if evidence.data_quality=='degraded':
            self.clean=0;self.latched=True;self.ack=False
            return self._result(2,'degraded_data')
        self.clean=self.clean+1 if contiguous else 1
        if self.latched and self.ack and self.clean>=self.required:self.latched=False
        if self.latched:return self._result(2,'recovery_review_pending')
        return self._result(1,'eligible_for_action_gate_review')

    def _latch(self,reason):
        self.latched=True;self.clean=0;self.ack=False
        return self._result(3,reason)

    def _result(self,level,reason):
        return {'schema_version':'D05RiskAdvisory/0.1','risk_level':level,
            'mode':{1:'request_review',2:'restricted',3:'takeover_requested'}[level],
            'automatic_execution_blocked':level!=1,'execution_authorized':False,
            'requires_action_gate':True,'reason':reason,'clean_observations':self.clean,
            'review_acknowledged':self.ack,'state_version':self.version,
            'episode_id':self.episode,'confidence_probability':None,
            'basis':'experimental evidence policy; no calibrated process reliability'}
