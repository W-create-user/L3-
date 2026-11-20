"""mAb model-scaled backend; imports optional MPCTools only at construction."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
from .base_adapter import ActionCoordinates
from .environment_adapter import EnvironmentAdapter, legacy_termination_flags, to_plain

class MAbAdapter:
    # Identical candidate validation, never silent clipping before D review.
    validate_action=EnvironmentAdapter.validate_action
    coerce_candidate=EnvironmentAdapter.coerce_candidate
    require_ready=EnvironmentAdapter.require_ready
    get_observation=EnvironmentAdapter.get_observation
    def __init__(self,source_dir=None,*,max_steps=60,initial_state_deviation_ratio=0.0):
        if type(max_steps) is not int or max_steps<=0:raise ValueError('max_steps must be positive')
        if not np.isfinite(initial_state_deviation_ratio) or not 0<=initial_state_deviation_ratio<=0.1:
            raise ValueError('Initial deviation must be in [0,0.1]')
        root=Path(__file__).resolve().parents[2]
        source=Path(source_dir or os.environ.get('SMPL_SOURCE_DIR') or root/'vendor/smpl').resolve()
        os.environ.setdefault('MPLCONFIGDIR',str(root/'.mpl-cache'));sys.path.insert(0,str(source))
        import mpctools
        if not all(hasattr(mpctools,name) for name in ['DiscreteSimulator','getCasadiFunc','nmpc']):
            raise ImportError('Install SMPL mpc-tools-casadi source, not the unrelated same-name package')
        from smpl.envs.mabenv import MAbEnvGym
        from smpl.envs.helpers.mab_helpers import xscale,uscale
        self._xscale=xscale.copy();self._uscale=uscale.copy();self.source_dir=str(source)
        self._env=MAbEnvGym(dataset_dir=str(source/'smpl/configdata/mabenv'),normalize=False,
                            max_steps=max_steps,initial_state_deviation_ratio=initial_state_deviation_ratio)
        self._observation=None;self._ready=False;self._done=False;self._version=0;self._step_calls=0
        self._closed=False;self._episode_id=None
    @property
    def metadata(self):
        return {'environment':'SMPL.MAbEnvGym','environment_id':'MAbEnvGym','normalize':False,
                'uss_subtracted':False,'action_coordinate':'physical_divided_by_uscale',
                'observation_coordinate':'physical_divided_by_xscale',
                'observation_return':'source_clipped_to_model_Box_not_a_safety_limit','sampling_period':{'value':self._env.dt_spl,'unit':'min'},
                'model_units_source':'vendor/smpl/smpl/envs/helpers/mab_helpers.py',
                'physical_process_safety_verified':False, 'episode_id':self._episode_id,
                'adapter_schema_version':'B03/1.0',
                'termination_note':'Upstream checks max_steps before increment; adapter honors configured successful-call cap.'}
    @property
    def action_space(self):return self._env.action_space
    @property
    def observation_space(self):return self._env.observation_space
    @property
    def action_coordinates(self):
        return ActionCoordinates(self.action_space.low,self.action_space.high,self._uscale,np.zeros(9))
    @property
    def state_version(self):return self._version
    @property
    def step_count(self):return int(self._env.step_count)
    @property
    def step_calls(self):return self._step_calls
    @property
    def done(self):return self._done
    def reset(self,seed=1101,*,initial_state=None):
        if self._closed:raise RuntimeError('Adapter closed')
        self._version+=1;self._ready=False;self._observation=None;self._episode_id=str(uuid4());np.random.seed(seed)
        if initial_state is not None:
            state=np.asarray(initial_state)
            if state.dtype.kind not in 'fiu':raise ValueError('Initial state must be numeric')
            with np.errstate(over='ignore',invalid='ignore'):
                state=state.astype(self.observation_space.dtype)
            if (state.shape!=self.observation_space.shape or state.dtype.kind not in 'fiu'
                    or not np.isfinite(state).all() or np.any(state<self.observation_space.low)
                    or np.any(state>self.observation_space.high)):
                raise ValueError('Explicit initial state must lie in live model bounds')
        obs=np.asarray(self._env.reset(initial_state=initial_state))
        if obs.shape!=self.observation_space.shape or not np.isfinite(obs).all():raise RuntimeError('Invalid mAb reset')
        self._observation=obs.copy();self._done=False;self._ready=True
        return self.get_observation()
    def step(self,action):
        self.require_ready();action=self.validate_action(action);before=self.step_count
        self._step_calls+=1;self._version+=1;self._ready=False
        # Uncertain exceptions leave this backend unready; never retry automatically.
        obs,reward,done,info=self._env.step(action)
        obs=np.asarray(obs);valid=obs.shape==self.observation_space.shape and np.isfinite(obs).all()
        raw=np.asarray(self._env.previous_observation)
        raw_valid=raw.shape==self.observation_space.shape and np.isfinite(raw).all()
        success=bool(valid and raw_valid and np.isfinite(reward) and float(reward)!=float(self._env.error_reward) and self.step_count==before+1)
        info=to_plain(info);info['upstream_done']=bool(done)
        info['clipped_indices']=np.flatnonzero(raw!=obs).tolist() if raw_valid and valid else None
        # Explicit adapter horizon fixes the documented upstream off-by-one without altering vendor code.
        cap=success and self.step_count>=self._env.max_steps
        if cap and not done:info.update(adapter_step_budget_reached=True,timeout=True)
        done=bool(done or cap or not success)
        terminated,truncated=legacy_termination_flags(done,info)
        self._observation=obs.copy() if valid else None;self._done=done;self._ready=success and not done
        return {'success':success,'observation':self.get_observation() if valid else None,'action':action.tolist(),
                'reward':float(reward) if np.isfinite(reward) else None,'done':done,
                'terminated':terminated,'truncated':truncated,'termination_flag_source':'SMPL_info_plus_explicit_adapter_horizon',
                'info':info,'step_count':self.step_count,'step_calls':self.step_calls,'state_version':self.state_version,
                'raw_preclip_observation':raw.tolist() if raw_valid else None,
                'failure_reason':None if success else 'mAb error reward, invalid observation or missing advance'}
    def close(self):
        self._closed=True;self._ready=False
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()
