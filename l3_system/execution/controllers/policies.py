"""Candidate-only control policies; no policy may step the environment."""
import numpy as np
from ..local_policy import space_midpoint

class SpaceMidpointPolicy:
    def reset(self):pass
    def predict(self,observation,action_space):return space_midpoint(observation,action_space)
    def commit(self,executed):pass

class BufferLevelPID:
    """Research-only physical h[dm] -> outlet[L/min]; positive error increases drain.

    Integral/derivative state commits only after a successful execution. Other axes
    use the source steady vector; no unverified capture-switch policy is added.
    """
    def __init__(self,adapter,config):
        if adapter.metadata.get('environment_id')!='MAbEnvGym':raise ValueError('mAb backend required')
        self.adapter=adapter;self.config=dict(config)
        for key in ['setpoint_dm','kp_L_min_per_dm','ki_L_min_per_dm_min','kd_L_min_min_per_dm','sampling_period_min']:
            value=config[key]
            if isinstance(value,bool) or not np.isfinite(value):raise ValueError('Invalid controller parameter')
        if any(config[key]<0 for key in ['kp_L_min_per_dm','ki_L_min_per_dm_min','kd_L_min_min_per_dm']):raise ValueError('Nonnegative gains required')
        if config['sampling_period_min']!=adapter.metadata['sampling_period']['value']:raise ValueError('Sampling mismatch')
        physical_low=adapter.observation_space.low[17]*adapter._xscale[17]
        physical_high=adapter.observation_space.high[17]*adapter._xscale[17]
        if not physical_low<=config['setpoint_dm']<=physical_high:raise ValueError('Setpoint outside model domain')
        self.reset()
    def reset(self):
        self.integral=0.;self.previous_error=None;self.saturation_count=0;self._pending=None
    def predict(self,observation,action_space):
        if self._pending is not None:raise RuntimeError('Commit or reject previous candidate first')
        obs=np.asarray(observation)
        if obs.shape!=(1970,) or not np.isfinite(obs).all():raise ValueError('Invalid mAb observation')
        if (tuple(action_space.shape)!=(9,) or not np.array_equal(action_space.low,self.adapter.action_space.low)
                or not np.array_equal(action_space.high,self.adapter.action_space.high)):raise ValueError('Action-space mismatch')
        c=self.config;dt=c['sampling_period_min'];error=float(obs[17]*self.adapter._xscale[17]-c['setpoint_dm'])
        integral=self.integral+error*dt
        derivative=0. if self.previous_error is None else (error-self.previous_error)/dt
        baseline=self.adapter.action_coordinates.to_physical(self.adapter._env.steady_actions)
        kp,ki,kd=[c[key] for key in ['kp_L_min_per_dm','ki_L_min_per_dm_min','kd_L_min_min_per_dm']]
        low=self.adapter.action_coordinates.to_physical(action_space.low)[7]
        high=self.adapter.action_coordinates.to_physical(action_space.high)[7]
        raw=baseline[7]+kp*error+ki*integral+kd*derivative
        # Conditional integration: discard integration that pushes farther into saturation.
        if (raw>high and error>0) or (raw<low and error<0):
            integral=self.integral;raw=baseline[7]+kp*error+ki*integral+kd*derivative
        clipped=float(np.clip(raw,low,high));baseline[7]=clipped
        action=self.adapter.action_coordinates.from_physical(baseline).astype(action_space.dtype)
        action=self.adapter.validate_action(action)
        self._pending=(integral,error,raw!=clipped)
        return action
    def commit(self,executed):
        if type(executed) is not bool:raise ValueError('Executed flag must be boolean')
        if self._pending is None:raise RuntimeError('No pending proposal')
        if executed:
            self.integral,self.previous_error,saturated=self._pending
            self.saturation_count+=int(saturated)
        self._pending=None
