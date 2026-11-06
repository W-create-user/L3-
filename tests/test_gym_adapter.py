import unittest
from types import SimpleNamespace
import numpy as np
from runtime.gym_adapter import GymEnvironmentAdapter


class Space:
    shape = (2,)
    dtype = np.dtype('float64')
    low, high = np.array([-1., -1.]), np.array([1., 1.])
    def contains(self, value):
        return value.shape == self.shape and bool(np.all(value >= self.low) and np.all(value <= self.high))


class GymAdapterTests(unittest.TestCase):
    def test_modern_termination_and_reset(self):
        env = SimpleNamespace(action_space=Space(), observation_space=Space(),
            reset=lambda seed: (np.zeros(2), {}),
            step=lambda action: (np.zeros(2), 1., False, True, {}), close=lambda: None)
        adapter = GymEnvironmentAdapter(env, environment_name='fixture', source_dir='.')
        adapter.reset()
        result = adapter.step([0., 0.])
        self.assertTrue(result['success'])
        self.assertTrue(result['truncated'])
        with self.assertRaises(RuntimeError):
            adapter.step([0., 0.])

    def test_legacy_reset_and_step(self):
        env = SimpleNamespace(action_space=Space(), observation_space=Space(),
            reset=lambda: np.zeros(2), step=lambda action: (np.zeros(2), 1., False, {}), close=lambda: None)
        adapter = GymEnvironmentAdapter(env, environment_name='fixture', source_dir='.')
        adapter.reset()
        self.assertTrue(adapter.step([0., 0.])['success'])
