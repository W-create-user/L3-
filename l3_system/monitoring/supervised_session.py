"""D05 simulation supervisor composed with unchanged A/B/C implementations.

No hidden warmup/recovery steps: each restricted sampling cycle is an explicit
operator request, uses midpoint and deterministic D review, and is recorded.
After a serious risk, only an explicit NEW episode can supply fresh simulator
states; repeating a frozen observation cannot satisfy recovery hysteresis.
"""
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
from uuid import uuid4
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from .risk_policy import RiskPolicy,RiskEvidence


class ObservedBackend:
    def __init__(self,base,on_reset,on_step):
        self.base=base;self.on_reset=on_reset;self.on_step=on_step
    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self,*args,**kwargs):
        result=self.base.reset(*args,**kwargs);self.on_reset();return result
    def step(self,action):
        result=self.base.step(action);self.on_step(result);return result


class SupervisedMonitor(MonitoringAgent):
    def __init__(self,supervisor,*args,**kwargs):
        self.supervisor=supervisor;super().__init__(*args,**kwargs)
    def check_action(self,state,candidate_action,task_id=None,*,process_context=None):
        result=super().check_action(state,candidate_action,task_id,process_context=process_context)
        return self.supervisor.gate(state,result,candidate_action)
    def check_observation(self,observation,task_id=None,*,process_context=None):
        result=super().check_observation(observation,task_id,process_context=process_context)
        return self.supervisor.gate(observation,result,None)


class SupervisedSession:
    def __init__(self,backend,model,output,*,fixture=False,clock=time.monotonic):
        self.base=backend;self.model=model;self.fixture=fixture;self.clock=clock
        if not fixture:
            binding=model.binding
            adapter_source=Path(__file__).resolve().parents[1]/'execution/environment_adapter.py'
            if (backend.action_space.shape!=(4,) or backend.observation_space.shape!=(39,)
                    or binding.get('environment')!='AtropineEnvGym' or binding.get('controller')!='space_midpoint'
                    or hashlib.sha256(adapter_source.read_bytes()).hexdigest()!=binding.get('adapter_sha256')):
                raise ValueError('live environment, controller or adapter/model binding differs')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.configuration=self._configuration()
        self.policy=RiskPolicy();self.window=[];self.last=None;self.observed_at=None;self.faulted=False
        self.manual_sampling=False;self.samples=0;self.events=[];self.episode=None
        self.proxy=ObservedBackend(backend,self._reset,self._sample)
        self.monitor=SupervisedMonitor(self,backend.action_space,backend.observation_space)
        self.executor=ExecutionAgent(self.proxy,log_path=self.output/'B_events.jsonl')
        self.system=MultiAgentSystem(executor=self.executor,monitor=self.monitor,event_path=self.output/'A_events.jsonl')

    def event(self,event,**fields):
        record={'event':event,'timestamp':datetime.now(timezone.utc).isoformat(),'monotonic_ns':time.perf_counter_ns(),
                'episode_id':self.episode,'state_version':self.base.state_version,**fields}
        self.events.append(record)
        with (self.output/'D_supervision.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')

    def _reset(self):
        self.episode=str(uuid4());self.policy.reset(self.episode);self.window=[];self.samples=0
        self.faulted=False;self.observed_at=self.clock();self.last=None
        self.captured=np.array(self.base.get_observation(),dtype=float);self.version=self.base.state_version
        self.event('new_episode_restricted',previous_history_reused=False)

    def start(self,seed=1101):self.executor.reset(seed=seed)

    def _configuration(self):
        env=getattr(self.base,'_env',None)
        flags={name:getattr(env,name,None) for name in ('normalize','uss_subtracted','observation_include_t',
            'observation_include_action','observation_include_uss','observation_include_ess',
            'observation_include_e','observation_include_kf','observation_include_z','observation_include_x')}
        return {'source_dir':str(getattr(self.base,'source_dir','fixture')),
                'action_low':np.asarray(self.base.action_space.low).tolist(),
                'action_high':np.asarray(self.base.action_space.high).tolist(),
                'observation_shape':list(self.base.observation_space.shape),'flags':flags}

    def _current(self,state):
        try:self.base.require_ready()
        except RuntimeError:return False
        return (self._configuration()==self.configuration
            and self.observed_at is not None and 0<=self.clock()-self.observed_at<=60
            and self.base.state_version==self.version and not self.base.done
            and np.array_equal(np.asarray(state),self.captured)
            and np.array_equal(np.asarray(self.base.get_observation()),self.captured))

    def _fault(self,reason):
        self.faulted=True;self.last=self.policy._latch(reason)
        self.event('takeover_requested',reason=reason)

    def _sample(self,result):
        # The B adapter returns success/state_version; executed is added later
        # by ExecutionAgent and is not part of the adapter-step contract.
        if result.get('success') is not True:
            self._fault('unknown_execution_result');return
        current=np.asarray(result['observation'],dtype=float)
        if (current.shape!=(39,) or not np.isfinite(current).all()
                or self.base.state_version!=self.version+1):
            self._fault('invalid_or_nonconsecutive_live_state');return
        self.version=self.base.state_version;self.captured=current.copy();self.observed_at=self.clock()
        self.window.append(current.copy());self.window=self.window[-4:];self.samples+=1
        if len(self.window)<4:
            self.event('restricted_warmup',available_samples=len(self.window));return
        try:
            score=self.model.score(np.array(self.window))
            evidence=RiskEvidence(self.episode,self.version,True,'valid',True,score['anomalous'],
                'in_scope' if score['in_empirical_reference_domain'] else 'out_of_scope')
            self.last=self.policy.evaluate(evidence)
            if self.last['risk_level']==3:self.faulted=True
            self.event('risk_evaluated',risk=self.last,model=score)
        except Exception:
            self._fault('model_or_scope_unavailable');raise

    def gate(self,state,result,action):
        if not result['approved']:
            self._fault(result['rule_id'] or 'hard_rule_rejected');return result
        reason=None
        if not self._current(state):reason='stale_changed_or_terminal_state'
        elif self.faulted:reason='takeover_latched'
        elif self.manual_sampling:
            midpoint=(self.base.action_space.low+self.base.action_space.high)/2
            if action is not None and not np.array_equal(np.asarray(action),midpoint):reason='restricted_sampling_midpoint_only'
        elif self.last is None or self.last['risk_level']!=1:reason='manual_review_or_valid_history_required'
        if reason:
            if reason=='stale_changed_or_terminal_state':self._fault(reason)
            result.update(approved=False,status='Reject',level=3 if self.faulted else 2,
                rule_id='R-03-D05',intervention='BLOCK',reason=reason)
        return result

    def sample(self,*,operator,reason):
        self._operator(operator,reason)
        if self.faulted or self.samples>=6 or self.system.state!='autonomous':
            raise RuntimeError('restricted sampling exhausted or serious risk; start reviewed new episode')
        self.event('explicit_restricted_sampling',operator=operator,reason=reason)
        self.manual_sampling=True
        try:
            result=self.system.run_cycle()
            if self.system.state!='autonomous':self._fault(result['summary']['failure_reason'] or 'session_stopped')
            return result
        except Exception:
            self._fault('sampling_execution_exception');raise
        finally:self.manual_sampling=False

    @staticmethod
    def _operator(operator,reason):
        if any(not isinstance(x,str) or not x.strip() for x in (operator,reason)):
            raise ValueError('operator and reason required')

    def confirm(self,*,operator,reason):
        self._operator(operator,reason)
        if self.policy.ack and not self.policy.latched:
            raise RuntimeError('review already confirmed')
        if (self.faulted or self.system.scheduler.active_agent_tasks or self.executor._pending
                or not self._current(self.base.get_observation()) or self.last is None):
            raise RuntimeError('unresolved risk, data or resources')
        before=self.base.step_calls
        self.last=self.policy.confirm_recovery(self.episode,state_version=self.version,reviewer_ref=operator+': '+reason)
        self.event('human_confirmed',operator=operator,reason=reason,step_delta=self.base.step_calls-before)
        return self.status()

    def cycle(self):
        if self.last is None or self.last['risk_level']!=1 or self.faulted:
            raise RuntimeError('reviewed D05 evidence required')
        try:
            result=self.system.run_cycle()
            if self.system.state!='autonomous':self._fault(result['summary']['failure_reason'] or 'session_stopped')
            return result
        except Exception:
            self._fault('autonomous_execution_exception');raise

    def reject_probe(self):
        if self.faulted:raise RuntimeError('already stopped')
        return self.system.run_cycle(action_generator=lambda o,s,a:s.high+1)

    def restart(self,*,operator,reason,seed=1101):
        self._operator(operator,reason)
        if self.system.scheduler.active_agent_tasks or self.executor._pending:
            raise RuntimeError('resolve resources before new episode')
        self.event('human_requested_new_episode',operator=operator,reason=reason)
        # Reconstruct A's session through its public constructor; no mutation of
        # another member's state machine or hidden resume/reset execution.
        self.executor.reset(seed=seed)
        self.system=MultiAgentSystem(executor=self.executor,monitor=self.monitor,event_path=self.output/'A_events.jsonl')
        return self.status()

    def status(self):
        return {'episode_id':self.episode,'state_version':self.base.state_version,'step_calls':self.base.step_calls,
            'mode':'takeover_requested' if self.faulted else self.last['mode'] if self.last else 'restricted',
            'risk':self.last,'samples':self.samples,'model_scope':'artificial sensor bias, empirical domain only',
            'physical_safety_certified':False,'human_factor_study':False}
