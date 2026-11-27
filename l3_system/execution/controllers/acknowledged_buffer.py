"""Opt-in B research controller with a trusted-startup balance estimate.

Implements the D25 observer idea with B's reset/predict/acknowledge/commit
transaction. No label, fault schedule or post-step scoring state is consumed.
Not identifiable under arbitrary initial bias or unacknowledged flow changes.
"""
import copy
import math
import numpy as np
from .policies import BufferLevelPID


class AcknowledgedBufferPolicy:
    def __init__(self, adapter, config, *, startup_level_dm, startup_trusted,
                 episode_id, state_version, allow_research=False,
                 max_innovation_dm=.3):
        if allow_research is not True or startup_trusted is not True:
            raise ValueError('Explicit research opt-in and trusted pre-fault startup required')
        if (not isinstance(episode_id, str) or not episode_id
                or type(state_version) is not int or state_version < 0
                or isinstance(startup_level_dm, bool) or not np.isfinite(startup_level_dm)
                or not 0 <= startup_level_dm <= 50
                or isinstance(max_innovation_dm, bool) or not np.isfinite(max_innovation_dm)
                or max_innovation_dm <= 0):
            raise ValueError('Invalid anchor or experimental innovation limit')
        self.adapter = adapter
        self.config = copy.deepcopy(config)
        self.base = BufferLevelPID(adapter, self.config)
        self.anchor = (float(startup_level_dm), episode_id, state_version)
        self.limit = float(max_innovation_dm)
        self.dt = float(config['sampling_period_min'])
        self.area = math.pi * 2.5**2
        self.events = []
        self.reset()

    def _fail(self, reason):
        self.failed = True
        raise RuntimeError(reason)

    def _context(self):
        return self.adapter.metadata['episode_id'], self.adapter.state_version

    def reset(self):
        if self._context() != self.anchor[1:]:
            raise RuntimeError('New episode requires a newly supplied trusted anchor')
        self.level, self.episode, self.version = self.anchor
        self.pending = None
        self.receipt = None
        self.failed = False
        self.events = []
        self.base.reset()

    @property
    def saturation_count(self):
        return self.base.saturation_count

    def predict(self, observation, action_space):
        if self.failed or self.pending is not None or self._context() != (self.episode, self.version):
            self._fail('Stale, pending or failed controller; no automatic retry')
        raw = np.asarray(observation, dtype=float)
        if raw.shape != (1970,) or not np.isfinite(raw).all():
            self._fail('Missing or nonfinite measurement')
        measurement = float(raw[17] * self.adapter._xscale[17])
        innovation = measurement-self.level
        if not 0 <= measurement <= 50 or abs(innovation) > self.limit:
            self._fail('Measurement/model mismatch outside declared experiment')
        delivered = raw.copy()
        delivered[17] = self.level / self.adapter._xscale[17]
        try:
            action = self.base.predict(delivered, action_space)
        except Exception:
            self.failed = True
            raise
        self.pending = np.asarray(action).copy()
        self.receipt = None
        self.events.append({'state_version': self.version, 'measurement_dm': measurement,
            'estimated_level_dm': self.level, 'innovation_dm': innovation,
            'basis': 'trusted startup plus acknowledged executed flow balance',
            'physical_safety_certified': False})
        return action

    def acknowledge(self, result):
        """Called exactly once by sole adapter owner, after actual step returns."""
        if self.failed or self.pending is None or self.receipt is not None:
            self._fail('Duplicate or unmatched execution receipt')
        if not isinstance(result, dict) or result.get('success') is not True:
            self._fail('Failed or unknown execution cannot advance observer')
        action = np.asarray(result.get('action'), dtype=float)
        if (self._context() != (self.episode, self.version+1)
                or result.get('state_version') != self.version+1
                or action.shape != (9,) or not np.isfinite(action).all()
                or not np.array_equal(action, self.pending)):
            self._fail('Receipt action/version/episode mismatch')
        # Store only acknowledged inputs, never the returned observation/reward.
        self.receipt = self.adapter.action_coordinates.to_physical(action).copy()

    def commit(self, executed):
        if type(executed) is not bool or self.failed or self.pending is None:
            self._fail('No valid pending control transaction')
        if not executed:
            if self.receipt is not None or self._context() != (self.episode, self.version):
                self._fail('Physical step occurred but chain did not confirm; takeover required')
            self.base.commit(False)
        else:
            if self.receipt is None or self._context() != (self.episode, self.version+1):
                self._fail('Successful chain requires matching executed-action receipt')
            level = self.level+self.dt*(self.receipt[3]-self.receipt[7])/self.area
            if not np.isfinite(level) or not 0 <= level <= 50:
                self._fail('Predicted state outside source model domain')
            self.base.commit(True)
            self.level = float(level)
            self.version += 1
        self.pending = None
        self.receipt = None
