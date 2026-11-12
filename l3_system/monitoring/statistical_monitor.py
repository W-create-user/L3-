"""Explicit experimental statistical R-03; cannot overrule deterministic rejection."""
import numpy as np
from l3_system.agents.monitoring_agent import MonitoringAgent


class StatisticalMonitoringAgent(MonitoringAgent):
    def __init__(self,*args,detector,window_provider,enable_experimental=False,**kwargs):
        if enable_experimental is not True:raise ValueError('statistical R-03 requires explicit experimental opt-in')
        if detector.threshold is None or not callable(window_provider):raise ValueError('trained detector and trusted window provider required')
        super().__init__(*args,**kwargs)
        if self.observation_shape!=(detector.dimension,):raise ValueError('detector/live observation dimensions differ')
        self.detector=detector;self.window_provider=window_provider

    def check_request(self,action_request,observation=None):
        decision=super().check_request(action_request,observation)
        if decision['decision']!='pass':return decision
        try:
            window=np.asarray(self.window_provider())
            if window.shape!=(self.detector.window_size,self.detector.dimension) or not np.array_equal(window[-1],observation):
                raise ValueError('window is stale or insufficient')
            scored=self.detector.score(window)
            reason='Statistical anomaly; request review, not a certified process hazard'
        except Exception as exc:
            scored={'error_type':type(exc).__name__};reason='Statistical context unavailable; execution blocked'
        if scored.get('anomalous',True):
            self.logs[-1].update(approved=False,status='Reject',level=3,rule_id='R-03-STAT',
                                 intervention='BLOCK',reason=reason,statistical_result=scored)
            return {**decision,'decision':'reject','rule_id':'R-03-STAT','reason':reason,
                    'risk_level':3,'statistical_result':scored,'basis':'validation-calibrated anomaly score, not physical safety limit'}
        self.logs[-1]['statistical_result']=scored
        return {**decision,'statistical_result':scored}
