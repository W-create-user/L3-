"""Explicit experimental statistical R-03; cannot overrule deterministic rejection."""
from copy import deepcopy
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent
from .binding import check_window


class StatisticalMonitoringAgent(MonitoringAgent):
    def __init__(self,*args,detector,window_provider,enable_experimental=False,trusted_context_provider=None,**kwargs):
        if enable_experimental is not True:raise ValueError('statistical R-03 requires explicit experimental opt-in')
        if detector.threshold is None or not callable(window_provider):raise ValueError('trained detector and trusted window provider required')
        super().__init__(*args,**kwargs)
        if self.observation_shape!=(detector.dimension,):raise ValueError('detector/live observation dimensions differ')
        self.detector=detector;self.window_provider=window_provider
        self.trusted_context_provider=trusted_context_provider
        self.binding=getattr(detector,'binding',None)
        if detector.label_scope!='synthetic_fixture' and self.binding is None:raise ValueError('nonfixture deployment requires bound model')
        if self.binding is not None and not callable(trusted_context_provider):raise ValueError('bound model requires trusted deployment context')
        self.statistical_r03_enabled=True

    def _statistical_gate(self,observation,result):
        if not result['approved']:return result
        try:
            packet=self.window_provider()
            if self.binding is not None:
                packet=check_window(packet,self.trusted_context_provider(),self.binding,self.detector.window_size)
            window=np.asarray(packet)
            if window.shape!=(self.detector.window_size,self.detector.dimension) or not np.array_equal(window[-1],observation):
                raise ValueError('window is stale or insufficient')
            scored=self.detector.score(window)
            if type(scored.get('anomalous')) is not bool:raise ValueError('invalid detector verdict')
            reason='Statistical anomaly; request review, not a certified process hazard'
        except Exception as exc:
            scored={'error_type':type(exc).__name__};reason='Statistical context unavailable; execution blocked'
        result['statistical_result']=scored
        if scored.get('anomalous',True):
            result.update(approved=False,status='Reject',level=3,rule_id='R-03-STAT',intervention='BLOCK',
                          reason=reason,basis='validation-calibrated anomaly score, not physical safety limit')
        return result

    def check_action(self,state,candidate_action,task_id=None,*,process_context=None):
        result=super().check_action(state,candidate_action,task_id,process_context=process_context)
        return self._statistical_gate(state,result)

    def check_observation(self,observation,task_id=None,*,process_context=None):
        result=super().check_observation(observation,task_id,process_context=process_context)
        return self._statistical_gate(observation,result)

    def check_request(self,action_request,observation=None):
        decision=super().check_request(action_request,observation)
        if self.binding is not None and decision['decision']=='pass':
            try:
                context=self.trusted_context_provider()
                if type(action_request.get('state_version')) is not int or action_request['state_version']!=context['state_version']:
                    raise ValueError('request state version differs from trusted live context')
            except Exception:
                reason='Request/live state version unavailable or mismatched'
                decision.update(decision='reject',rule_id='R-03-STAT',reason=reason,risk_level=3)
                self.logs[-1].update(approved=False,status='Reject',level=3,rule_id='R-03-STAT',reason=reason,intervention='BLOCK')
        if self.logs and 'statistical_result' in self.logs[-1]:
            decision['statistical_result']=deepcopy(self.logs[-1]['statistical_result'])
        return decision
