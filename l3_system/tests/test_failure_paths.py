"""Regression tests for failures between roles, using a counting test environment."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from l3_system.main import run_control_cycle
from l3_system.execution import ExecutionAgent
from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.algorithms.cnp_protocol import CNPError
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter


class FailurePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.adapter = FakeAdapter()
        self.executor = ExecutionAgent(self.adapter, Path(self.tmp.name)/'B.jsonl')
        self.monitor = MonitoringAgent(self.adapter.action_space)
        self.scheduler = SchedulingAgent()
        self.executor.reset()

    def run_cycle(self, **kwargs):
        return run_control_cycle(goal='执行一次控制任务', executor=self.executor,
            monitor=self.monitor, scheduler=self.scheduler, **kwargs)

    def assert_failed_without_locks(self, result, steps=0):
        self.assertEqual(result['summary']['run_status'], 'incomplete_or_failed')
        self.assertEqual(self.adapter.step_calls, steps)
        self.assertFalse(self.scheduler.active_agent_tasks)
        self.assertFalse(self.executor._pending)
        self.assertIn('failed', result['summary']['subtask_statuses'].values())
        self.assertIn('blocked', result['summary']['subtask_statuses'].values())
        self.assertEqual(result['parent_task']['status'], 'failed')

    def test_observation_exception_returns_failed_feedback(self):
        with patch.object(self.executor, 'get_observation', side_effect=RuntimeError('sensor unavailable')):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_generator_exception_releases_decision_lock(self):
        def fail(*args):
            raise RuntimeError('policy unavailable')
        self.assert_failed_without_locks(self.run_cycle(action_generator=fail))

    def test_nonfinite_candidate_is_blocked(self):
        self.assert_failed_without_locks(self.run_cycle(action_generator=lambda *args: [np.nan]*4))

    def test_monitor_exception_discards_pending_candidate(self):
        with patch.object(self.monitor, 'check_request', side_effect=RuntimeError('monitor offline')):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_missing_review_is_blocked(self):
        with patch.object(self.monitor, 'check_request', return_value=None):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_mismatched_task_review_is_blocked(self):
        with patch.object(self.monitor, 'check_request', return_value={'task_id':'wrong','decision':'pass'}):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_conflicting_review_is_blocked(self):
        def conflicting(request, observation):
            return {'task_id':request['task_id'], 'decision':'pass', 'approved':False}
        with patch.object(self.monitor, 'check_request', side_effect=conflicting):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_second_review_rejection_blocks_final_execution(self):
        count = 0
        original = self.monitor.check_request
        def changed(request, observation):
            nonlocal count
            count += 1
            if count == 1:
                return original(request, observation)
            return {'task_id':request['task_id'], 'decision':'reject', 'reason':'changed safety state', 'rule_id':'test'}
        with patch.object(self.monitor, 'check_request', side_effect=changed):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)
        self.assertEqual(count, 2)

    def test_environment_exception_releases_executor_lock(self):
        with patch.object(self.adapter, 'step', side_effect=RuntimeError('solver failure')):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_nonterminal_executor_result_cannot_complete_synchronous_cycle(self):
        def running(request, review):
            return {'task_id':request['task_id'],'subtask_id':request['subtask_id'],
                    'execution_status':'running'}
        with patch.object(self.executor, 'execute_action', side_effect=running):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_no_bids_fails_with_feedback(self):
        with patch('l3_system.main.configured_bids', return_value=[]):
            result = self.run_cycle()
        self.assert_failed_without_locks(result)

    def test_continuous_cycles_preserve_state_and_release_locks(self):
        ids = set()
        for _ in range(5):
            result = self.run_cycle()
            self.assertEqual(result['summary']['run_status'], 'completed')
            self.assertEqual(result['summary']['step_delta'], 1)
            self.assertTrue(result['summary']['all_C_locks_released'])
            ids.add(result['summary']['task_id'])
        self.assertEqual(len(ids), 5)
        self.assertEqual(self.adapter.step_calls, 5)

    def test_wrong_parent_feedback_cannot_release_other_task(self):
        self.scheduler.register_task({'task_id':'parent','subtask_id':'child', 'task_type':'read', 'status':'pending'})
        with self.assertRaises(CNPError):
            self.scheduler.update_feedback({'task_id':'wrong','subtask_id':'child','execution_status':'completed'})
        self.assertEqual(self.scheduler.get_task_status('parent','child'), 'pending')

    def test_wrong_parent_status_query_is_refused(self):
        self.scheduler.register_task({'task_id':'parent','subtask_id':'child', 'task_type':'read', 'status':'pending'})
        with self.assertRaises(CNPError):
            self.scheduler.get_task_status('wrong','child')

    def test_demo_review_cannot_authorize_low_level_execution(self):
        request = self.executor.prepare_action({'task_id':'test', 'parameters':{'action_policy':'space_midpoint'}})
        with self.assertRaises(ValueError):
            self.executor.execute_action(request, {'task_id':'test','decision':'pass','demo_mode':True})
        self.assertEqual(self.adapter.step_calls, 0)
