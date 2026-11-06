import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B_CODE = ROOT.parent / '执行智能体' / 'code'
sys.path.insert(0, str(B_CODE))

from integration_abd_1103.execution_agent import ExecutionAgent
from a_integration.agents.monitoring_agent import MonitoringAgent
from a_integration.main import run_control_cycle


class FakeBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.shape = self.low.shape
        self.dtype = np.dtype('float64')

    def contains(self, value):
        arr = np.asarray(value)
        return (arr.shape == self.shape and np.isfinite(arr).all()
                and np.all(arr >= self.low) and np.all(arr <= self.high))


class FakeAdapter:
    """Test-only adapter; it is not SMPL and is never counted as a live run."""
    def __init__(self):
        self.action_space = FakeBox([-1.0] * 4, [1.0] * 4)
        self.observation_space = SimpleNamespace(shape=(39,))
        self.source_dir = 'TEST_FAKE_ADAPTER_NOT_SMPL'
        self.state_version = 0
        self.step_count = 0
        self.step_calls = 0
        self.done = False
        self._obs = np.zeros(39, dtype=float)
        self._ready = True

    def reset(self, seed=1101):
        self._obs = np.zeros(39, dtype=float)
        self._ready = True
        self.done = False
        return self.get_observation()

    def require_ready(self):
        if not self._ready:
            raise RuntimeError('fake adapter not ready')

    def get_observation(self):
        return self._obs.copy().tolist()

    def coerce_candidate(self, action):
        arr = np.asarray(action, dtype=float)
        if arr.shape != self.action_space.shape or not np.isfinite(arr).all():
            raise ValueError('invalid fake candidate')
        return arr

    def validate_action(self, action):
        arr = self.coerce_candidate(action)
        if not self.action_space.contains(arr):
            raise ValueError('out of fake bounds')
        return arr

    def step(self, action):
        self.require_ready()
        arr = self.validate_action(action)
        self.step_calls += 1
        self.step_count += 1
        self.state_version += 1
        self._obs = self._obs + 0.01
        return {'success': True, 'observation': self.get_observation(),
                'action': arr.tolist(), 'reward': 0.25, 'done': False,
                'terminated': False, 'truncated': False, 'info': {},
                'step_count': self.step_count, 'step_calls': self.step_calls,
                'state_version': self.state_version, 'failure_reason': None}


class LiveOrchestrationContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.monitor = MonitoringAgent(action_space=self.adapter.action_space)
        self.temp = tempfile.TemporaryDirectory()
        self.executor = ExecutionAgent(self.adapter, log_path=Path(self.temp.name) / 'B.jsonl')
        self.executor.reset(seed=1101)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_six_subtask_chain_with_real_B_agent_and_D_monitor_fake_env(self):
        result = run_control_cycle(
            goal={'task_type': 'control_cycle', 'description': 'test cycle', 'priority': 3},
            executor=self.executor, monitor=self.monitor,
            event_path=Path(self.temp.name) / 'A.jsonl')
        self.assertEqual(result['summary']['run_status'], 'completed')
        self.assertEqual(result['summary']['completed_subtasks'], 6)
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertEqual(result['summary']['bids_are_live'], False)
        self.assertFalse(self.monitor.demo_mode)
        self.assertGreaterEqual(len(self.monitor.get_logs()), 2)  # HTN review + B final gate

    def test_real_D_preflight_reject_blocks_before_B_step(self):
        def out_of_range_action(observation, action_space, attempt=0):
            return action_space.high + 1.0
        result = run_control_cycle(
            goal={'task_type': 'control_cycle', 'description': 'reject cycle', 'priority': 3},
            executor=self.executor, monitor=self.monitor,
            action_generator=out_of_range_action)
        self.assertEqual(result['summary']['run_status'], 'incomplete_or_failed')
        self.assertEqual(self.adapter.step_calls, 0)
        rejected = [x for x in self.monitor.get_logs() if x['status'] == 'Reject']
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]['rule_id'], 'R-02')

    def test_inspect_goal_has_three_stages_and_no_step(self):
        result = run_control_cycle(goal='检查状态', executor=self.executor, monitor=self.monitor)
        self.assertEqual(result['summary']['run_status'], 'completed')
        self.assertEqual(result['summary']['completed_subtasks'], 3)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_demo_mode_is_refused(self):
        demo_monitor = MonitoringAgent(action_space=None)
        with self.assertRaises(RuntimeError):
            run_control_cycle(
                goal={'task_type': 'control_cycle', 'description': 'demo', 'priority': 3},
                executor=self.executor, monitor=demo_monitor)


if __name__ == '__main__':
    unittest.main()
