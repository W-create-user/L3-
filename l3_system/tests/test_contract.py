"""Fast contract checks with a counting stand-in; demo_ab covers real SMPL."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from l3_system.execution.execution_agent import ExecutionAgent


class Space:
    shape = (2,)
    dtype = np.dtype('float32')
    low = np.array([-1, -2], dtype=dtype)
    high = np.array([1, 2], dtype=dtype)


class CountingAdapter:
    source_dir = 'contract-test-stand-in-not-SMPL'
    action_space = Space()
    done = False
    step_count = 0
    step_calls = 0
    state_version = 0

    def reset(self, seed=1101):
        self.state_version += 1
        self.step_count = 0
        return self.get_observation()

    def require_ready(self):
        pass

    def get_observation(self):
        return [float(self.step_count)]

    def validate_action(self, action):
        raw = np.asarray(action)
        if (raw.shape != (2,) or raw.dtype.kind not in 'fiu'
                or not np.isfinite(raw).all()
                or np.any(raw < self.action_space.low)
                or np.any(raw > self.action_space.high)):
            raise ValueError('invalid action')
        return raw.astype(self.action_space.dtype)

    def coerce_candidate(self, action):
        raw = np.asarray(action)
        if (raw.shape != (2,) or raw.dtype.kind not in 'fiu'
                or not np.isfinite(raw).all()):
            raise ValueError('invalid candidate')
        return raw.astype(self.action_space.dtype)

    def step(self, action):
        self.validate_action(action)
        self.step_calls += 1
        self.step_count += 1
        self.state_version += 1
        return {'success': True, 'observation': self.get_observation(),
                'action': list(action), 'reward': 0.0, 'done': False, 'info': {},
                'step_count': self.step_count, 'step_calls': self.step_calls,
                'state_version': self.state_version, 'failure_reason': None}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = CountingAdapter()
        self.agent = ExecutionAgent(self.adapter, Path(self.temp.name) / 'events.jsonl')
        self.agent.reset()

    def task(self, parameters=None):
        return {'task_id': 'A-1', 'parameters': parameters or {'action_policy': 'space_midpoint'}}

    def review(self, request, decision='pass'):
        return {'request_id': request['request_id'], 'decision': decision, 'reason': 'unit test'}

    def test_preparation_never_steps(self):
        self.agent.get_observation()
        request = self.agent.prepare_action(self.task())
        self.assertEqual(request['action'], [0.0, 0.0])
        self.assertEqual(self.adapter.step_calls, 0)

    def test_explicit_task_action(self):
        request = self.agent.prepare_action(self.task({'action': [0.5, 1.0]}))
        result = self.agent.execute_action(request, self.review(request))
        self.assertEqual(result['task_id'], 'A-1')
        self.assertTrue(result['success'])
        self.assertEqual(self.adapter.step_calls, 1)

    def test_missing_review_never_steps(self):
        request = self.agent.prepare_action(self.task())
        with self.assertRaises(ValueError):
            self.agent.execute_action(request)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_rejection_never_steps(self):
        request = self.agent.prepare_action(self.task())
        result = self.agent.execute_action(request, self.review(request, 'reject'))
        self.assertEqual(result['execution_status'], 'rejected')
        self.assertFalse(result['executed'])
        self.assertEqual(self.adapter.step_calls, 0)

    def test_wrong_review_never_steps(self):
        request = self.agent.prepare_action(self.task())
        review = self.review(request)
        review['request_id'] = 'other'
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, review)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_changed_request_never_steps(self):
        request = self.agent.prepare_action(self.task())
        changed = copy.deepcopy(request)
        changed['action'][0] = 0.5
        with self.assertRaises(ValueError):
            self.agent.execute_action(changed, self.review(request))
        self.assertEqual(self.adapter.step_calls, 0)

    def test_duplicate_request_never_steps_twice(self):
        request = self.agent.prepare_action(self.task())
        review = self.review(request)
        self.agent.execute_action(request, review)
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, review)
        self.assertEqual(self.adapter.step_calls, 1)

    def test_state_change_invalidates_pending_request(self):
        request = self.agent.prepare_action(self.task())
        self.adapter.reset()
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, self.review(request))
        self.assertEqual(self.adapter.step_calls, 0)

    def test_reset_clears_pending_requests(self):
        request = self.agent.prepare_action(self.task())
        self.agent.reset()
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, self.review(request))

    def test_unsupported_task_does_not_invent_action(self):
        for task in [{'task_id': 'A-1'}, {'task_id': 'A-1', 'parameters': {'action_policy': 'unknown'}},
                     {'parameters': {'action': [0, 0]}}]:
            with self.subTest(task=task), self.assertRaises(ValueError):
                self.agent.prepare_action(task)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_invalid_actions_do_not_step(self):
        for action in [[0], [float('nan'), 0], ['x', 'y']]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.agent.prepare_action(self.task({'action': action}))
        self.assertEqual(self.adapter.step_calls, 0)


if __name__ == '__main__':
    unittest.main()
