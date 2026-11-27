"""D-owned development candidate around B's unchanged P controller.

Needs a trusted, pre-fault startup level. Uses only delivered measurement and
acknowledged executed flows. It cannot identify an arbitrary initial bias.
This is a model-mass-balance observer, not an industrial sensor estimator.
"""
import math
import numpy as np

class BufferObserver:
    def __init__(self,*,startup_level_dm,startup_trusted,episode_id,state_version=0,
                 dt_min=60.,area_dm2=math.pi*2.5**2,max_innovation_dm=.3):
        if startup_trusted is not True or not isinstance(episode_id,str) or not episode_id:
            raise ValueError('trusted pre-fault startup calibration required')
        if not np.isfinite(startup_level_dm) or not 0<=startup_level_dm<=50 or type(state_version)is not int or state_version<0:
            raise ValueError('invalid startup level/version')
        if not all(np.isfinite(x) and x>0 for x in (dt_min,area_dm2,max_innovation_dm)):raise ValueError('positive model parameters required')
        self.level=float(startup_level_dm);self.episode=episode_id;self.version=state_version
        self.dt=dt_min;self.area=area_dm2;self.limit=max_innovation_dm;self.pending=False;self.failed=False
    def propose(self,measurement_dm,*,episode_id,state_version):
        if self.failed or self.pending or type(state_version)is not int or episode_id!=self.episode or state_version!=self.version:
            self.failed=True;raise ValueError('stale, repeated or unknown observer state')
        if not np.isfinite(measurement_dm) or abs(measurement_dm-self.level)>self.limit:
            self.failed=True;raise ValueError('measurement/model disagreement outside experiment scope')
        self.pending=True
        return {'estimated_level_dm':self.level,'estimated_bias_dm':float(measurement_dm-self.level),
                'basis':'trusted startup plus acknowledged flow mass balance','physical_safety_certified':False}
    def commit(self,*,executed,physical_action,state_version):
        if type(executed)is not bool or not self.pending or self.failed:raise ValueError('unmatched observer commit')
        self.pending=False
        if not executed:
            if state_version!=self.version:self.failed=True;raise ValueError('rejected action advanced state')
            return
        u=np.asarray(physical_action,dtype=float)
        if u.shape!=(9,) or not np.isfinite(u).all() or state_version!=self.version+1:
            self.failed=True;raise ValueError('unknown/mismatched executed joint action')
        self.level+=self.dt*(u[3]-u[7])/self.area;self.version=state_version
        if not 0<=self.level<=50:self.failed=True;raise ValueError('observer left model domain')
