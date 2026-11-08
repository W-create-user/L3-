import tempfile
import unittest
from pathlib import Path

import numpy as np

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import ExecutionAgent
from l3_system.runtime import MultiAgentSystem
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter


class SystemRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = FakeAdapter()
        self.monitor = MonitoringAgent(self.adapter.action_space, self.adapter.observation_space)
        self.executor = ExecutionAgent(self.adapter, log_path=Path(self.temp.name)/'B.jsonl')
        self.system = MultiAgentSystem(executor=self.executor, monitor=self.monitor,
                                       event_path=Path(self.temp.name)/'events.jsonl')

    @staticmethod
    def bad(obs, space, attempt):
        return space.high + 1

    def test_session_preserves_all_four_roles_and_scheduler_feedback(self):
        for _ in range(2):
            result = self.system.run_cycle()
            self.assertEqual(result['summary']['system_state'], 'autonomous')
            self.assertEqual(result['summary']['completed_subtasks'], 6)
            self.assertTrue(all(e.get('task_id') for e in result['events']))
        self.assertEqual(self.adapter.step_calls, 2)
        self.assertEqual(len(self.system.scheduler.feedback_log), 12)
        self.assertFalse(self.system.scheduler.active_agent_tasks)

    def test_rejection_latches_until_explicit_human_resume(self):
        result = self.system.run_cycle(action_generator=self.bad)
        self.assertEqual(result['summary']['system_state'], 'takeover_requested')
        with self.assertRaises(RuntimeError):
            self.system.run_cycle()
        self.assertEqual(self.adapter.step_calls, 0)
        with self.assertRaises(ValueError):
            self.system.resume(operator='', reason='fixed')
        self.system.resume(operator='test operator', reason='replaced candidate policy')
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertEqual(self.system.run_cycle()['summary']['run_status'], 'completed')

    def test_bounded_regeneration_replans_without_replaying_a_step(self):
        self.system.max_action_regenerations = 1
        seen = []
        def proposal(obs, space, attempt):
            seen.append(attempt)
            return space.high + 1 if attempt == 0 else (space.low + space.high)/2
        result = self.system.run_cycle(action_generator=proposal)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(result['summary']['system_state'], 'autonomous')
        self.assertEqual(result['summary']['action_regenerations'], 1)
        self.assertEqual(result['recovery_attempts'][0]['summary']['step_delta'], 0)
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertFalse(self.executor._pending)

    def test_repeated_rejection_exhausts_limit_and_requests_takeover(self):
        self.system.max_action_regenerations = 2
        result = self.system.run_cycle(action_generator=self.bad)
        self.assertEqual(result['summary']['system_state'], 'takeover_requested')
        self.assertEqual(len(self.system.history), 3)
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(self.system.scheduler.active_agent_tasks)

    def test_uncertain_physical_failure_is_never_retried(self):
        def broken(action):
            self.adapter.step_calls += 1
            self.adapter._ready = False
            raise RuntimeError('solver failed after dispatch')
        self.adapter.step = broken
        self.system.max_action_regenerations = 3
        result = self.system.run_cycle(action_generator=lambda o, s, a: s.low)
        self.assertEqual(result['summary']['system_state'], 'takeover_requested')
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertEqual(len(self.system.history), 1)
        with self.assertRaises(RuntimeError):
            self.system.resume(operator='tester', reason='acknowledged')
        self.system.resume(operator='tester', reason='new simulation episode', reset=True)
        self.assertEqual(self.system.state, 'autonomous')
        self.assertEqual(self.adapter.step_calls, 1)

    def test_monitor_error_never_triggers_automatic_retry(self):
        def broken(*args):
            raise RuntimeError('checker unavailable')
        self.monitor.check_request = broken
        self.system.max_action_regenerations = 3
        result = self.system.run_cycle(action_generator=lambda o, s, a: s.low)
        self.assertEqual(result['summary']['failure_reason'], 'safety_review_error')
        self.assertEqual(len(self.system.history), 1)
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(self.executor._pending)

    def test_second_gate_rejection_is_reported_consistently(self):
        original = self.monitor.check_request
        calls = []
        def reject_late(*args):
            response = original(*args)
            calls.append(response)
            if len(calls) == 2:
                response.update(decision='reject', rule_id='test', reason='changed risk')
            return response
        self.monitor.check_request = reject_late
        result = self.system.run_cycle()
        self.assertEqual(result['summary']['failure_reason'], 'safety_rejected')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_invalid_observation_blocks_resume_and_action(self):
        self.system.run_cycle(action_generator=self.bad)
        self.adapter._obs[0] = np.nan
        with self.assertRaises(RuntimeError):
            self.system.resume(operator='tester', reason='acknowledged')
        decision = self.monitor.check_request({'task_id':'T', 'action':[0]*4}, [np.nan]*39)
        self.assertEqual(decision['decision'], 'reject')
        self.assertEqual(decision['rule_id'], 'STATE-INVALID')
        self.assertEqual(self.system.state, 'takeover_requested')

    def test_post_execution_observation_is_independently_monitored(self):
        original = self.adapter.step
        def corrupt(action):
            result = original(action)
            self.adapter._obs = np.zeros(2)
            return result
        self.adapter.step = corrupt
        result = self.system.run_cycle()
        self.assertEqual(result['summary']['failure_reason'], 'invalid_post_execution_observation')
        self.assertEqual(self.system.state, 'takeover_requested')
        self.assertEqual(self.adapter.step_calls, 1)

    def test_parent_dependency_is_not_silently_discarded(self):
        result = self.system.run_cycle({'task_type':'control_cycle', 'dependencies':['missing-parent']})
        self.assertEqual(result['summary']['run_status'], 'incomplete_or_failed')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_invalid_goal_requests_takeover_without_execution(self):
        with self.assertRaises(ValueError):
            self.system.run_cycle('unsupported goal')
        self.assertEqual(self.system.state, 'takeover_requested')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_unimplemented_process_constraints_are_not_reported_as_met(self):
        with self.assertRaises(NotImplementedError):
            self.system.run_cycle({'task_type':'control_cycle', 'constraints':{'yield_min':0.9}})
        self.assertEqual(self.system.state, 'takeover_requested')
        self.assertEqual(self.adapter.step_calls, 0)


if __name__ == '__main__':
    unittest.main()
