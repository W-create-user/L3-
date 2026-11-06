"""Tests against D's actual pinned PR source, with a counting simulator stand-in."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

from .demo_actual_a import load_monitor
from .execution_agent import ExecutionAgent
from .safety_bridge import normalize_safety_decision
from .test_contract import CountingAdapter


class ActualDContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = CountingAdapter()
        self.agent = ExecutionAgent(self.adapter, Path(self.temp.name) / 'events.jsonl')
        self.agent.reset()
        source = Path(os.environ.get('D_MONITORING_FILE') or
                      Path(__file__).resolve().parents[1] / 'external_D/pr1_8c105883/source/agents/monitoring_agent.py')
        self.monitor, provenance = load_monitor(self.adapter.action_space, source)
        self.assertEqual(provenance, 'D_provided_source')

    def request(self, action=None):
        if action is None:
            return self.agent.prepare_action({'task_id': 'D-PR1', 'task_type': 'control_cycle'})
        return self.agent.prepare_action({'task_id': 'D-PR1'}, action=action)

    def quiet(self, method, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return method(*args)

    def test_two_argument_check_action_null_task_id_is_bound(self):
        request = self.request()
        review = self.quiet(self.monitor.check_action, request['observation'], request['action'])
        self.assertIsNone(review['task_id'])
        result = self.agent.execute_action(request, review)
        self.assertTrue(result['success'])
        self.assertEqual(result['safety_decision']['task_id'], 'D-PR1')
        self.assertEqual(self.adapter.step_calls, 1)

    def test_check_request_without_request_id_is_compatible(self):
        request = self.request()
        review = self.quiet(self.monitor.check_request, request, request['observation'])
        self.assertNotIn('request_id', review)
        self.assertNotIn('approved', review)
        result = self.agent.execute_action(request, review)
        self.assertTrue(result['success'])
        self.assertEqual(result['safety_decision']['risk_level'], 0)
        self.assertEqual(result['safety_decision']['level'], 0)

    def test_actual_D_rejection_does_not_step(self):
        request = self.request([99, 0])
        review = self.quiet(self.monitor.check_request, request, request['observation'])
        result = self.agent.reject_action_request(request, review)
        self.assertEqual(result['safety_decision']['rule_id'], 'R-02')
        self.assertEqual(result['safety_decision']['risk_level'], 3)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_explicit_wrong_task_id_is_still_refused(self):
        request = self.request()
        review = {'approved': True, 'status': 'Pass', 'task_id': 'other-task'}
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, review)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_unbound_decision_without_any_id_is_refused(self):
        request = self.request()
        with self.assertRaises(ValueError):
            normalize_safety_decision(request, {'decision': 'pass'})


if __name__ == '__main__':
    unittest.main()
