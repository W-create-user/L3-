import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from runtime.cluster import AgentCluster
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


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = TestAdapter()
        self.cluster = AgentCluster(self.adapter, log_path=Path(self.temp.name) / 'events.jsonl')

    def test_full_cycle_preserves_identity_and_feedback(self):
        result = self.cluster.run('执行一次控制任务')
        self.assertTrue(result['success'])
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertEqual(len(result['events']), 6)
        self.assertEqual([e['assignment']['winner_id'] for e in result['events']],
                         ['B', 'A', 'A', 'D', 'B', 'A'])
        for sub, event in zip(result['task']['subtasks'], result['events']):
            self.assertEqual(event['feedback']['subtask_id'], sub['subtask_id'])
            self.assertEqual(event['assignment']['task_id'], result['task']['task_id'])
        self.assertFalse(self.cluster.scheduler.active_agent_tasks)

    def test_rejection_blocks_environment_and_dependents(self):
        result = self.cluster.run('执行一次控制任务', action=[2., 0.])
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'safety_rejected')
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(result['execution_result']['executed'])
        self.assertEqual(list(result['subtask_statuses'].values())[-2:], ['pending', 'pending'])
        self.assertFalse(self.cluster.scheduler.active_agent_tasks)

    def test_inspect_does_not_step(self):
        result = self.cluster.run('检查状态')
        self.assertTrue(result['success'])
        self.assertEqual(self.adapter.step_calls, 0)

    def test_execution_failure_stops_verification(self):
        self.adapter.fail = True
        result = self.cluster.run('执行一次控制任务')
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_reason'], 'simulator_failure')
        self.assertEqual(result['events'][-1]['name'], 'execute_approved_action')

    def test_repeated_runs_reset_scheduler(self):
        for _ in range(2):
            self.assertTrue(self.cluster.run('执行一次控制任务')['success'])
        self.assertEqual(self.adapter.step_calls, 2)

    def test_stale_and_consumed_requests_are_rejected(self):
        self.cluster.run('执行一次控制任务')
        task = self.cluster.decision.task_to_dict(self.cluster.decision.create_task('执行一次控制任务'))
        request = self.cluster.execution.prepare_action(task)
        self.cluster.execution.reset()
        with self.assertRaises(ValueError):
            self.cluster.execution.execute_action(request, {'approved': True})


if __name__ == '__main__':
    unittest.main()
