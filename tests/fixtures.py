from types import SimpleNamespace
import numpy as np
from runtime.environment_adapter import EnvironmentAdapter

class TestAdapter(EnvironmentAdapter):
    """Deterministic interface fixture, never presented as the SMPL simulator."""
    def __init__(self):
        self.source_dir = 'test-fixture'
        self._space = SimpleNamespace(shape=(2,), low=np.array([-1., -1.]),
                                      high=np.array([1., 1.]), dtype=np.dtype('float64'))
        self._version = self._step_calls = 0
        self._ready = False
        self._done = False
        self._observation = None
        self.fail = False

    @property
    def action_space(self):
        return self._space

    @property
    def step_count(self):
        return self._step_calls

    def reset(self, seed=1101):
        self._version += 1
        self._observation = np.array([0., 0.])
        self._ready = True
        self._done = False
        return self.get_observation()

    def step(self, action):
        self.require_ready()
        action = np.asarray(action)
        assert np.all(action >= self._space.low) and np.all(action <= self._space.high)
        self._version += 1
        self._step_calls += 1
        self._observation = action.copy()
        return {'success': not self.fail, 'observation': action.tolist(),
                'done': False, 'failure_reason': 'simulator_failure' if self.fail else None}


