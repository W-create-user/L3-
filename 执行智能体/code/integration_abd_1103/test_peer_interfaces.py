import copy
from pathlib import Path
import tempfile
import unittest

from .a_integration import load_decision_agent, run_a_plan
from .execution_agent import ExecutionAgent
from .reference_monitor import ReferenceMonitoringAgent
from .safety_bridge import normalize_safety_decision
from .test_contract import CountingAdapter


class PeerInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = CountingAdapter()
        self.agent = ExecutionAgent(self.adapter, Path(self.temp.name) / 'B.jsonl')
        self.agent.reset()
        self.monitor = ReferenceMonitoringAgent(self.adapter.action_space)

    def task(self):
        return {'task_id': 'actual-shape-A', 'goal_type': 'control_cycle',
                'subtasks': [{'subtask_id': 'SUB-exec', 'name': 'execute_approved_action'}]}

    def test_complete_A_task_with_explicit_B_policy_is_not_mutated(self):
        task = self.task()
        saved = copy.deepcopy(task)
        request = self.agent.prepare_action(task, action_policy='space_midpoint', subtask=task['subtasks'][0])
        self.assertEqual(task, saved)
        self.assertEqual(request['subtask_id'], 'SUB-exec')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_raw_D_approved_result_can_execute(self):
        request = self.agent.prepare_action(self.task(), action_policy='space_midpoint')
        raw = self.monitor.check_action(self.agent.get_observation(), request['action'])
        result = self.agent.execute_action(request, raw)
        self.assertTrue(result['success'])
        self.assertEqual(result['safety_decision']['raw_review']['status'], 'Pass')
        self.assertEqual(self.adapter.step_calls, 1)

    def test_out_of_bounds_reaches_D_and_is_rejected_without_step(self):
        request = self.agent.prepare_action(self.task(), action=[99, 0])
        raw = self.monitor.check_action(self.agent.get_observation(), request['action'])
        result = self.agent.reject_action_request(request, raw)
        self.assertEqual(result['safety_decision']['rule_id'], 'R-02')
        self.assertEqual(result['execution_status'], 'rejected')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_even_an_approved_out_of_bounds_candidate_cannot_step(self):
        request = self.agent.prepare_action(self.task(), action=[99, 0])
        with self.assertRaises(ValueError):
            self.agent.execute_action(request, {'approved': True, 'status': 'Pass'})
        self.assertEqual(self.adapter.step_calls, 0)

    def test_approved_strings_are_not_truthy_approvals(self):
        request = self.agent.prepare_action(self.task(), action_policy='space_midpoint')
        for value in ['False', 'True', 1, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.agent.execute_action(request, {'approved': value})
        self.assertEqual(self.adapter.step_calls, 0)

    def test_conflicting_monitor_status_is_refused(self):
        request = self.agent.prepare_action(self.task(), action_policy='space_midpoint')
        with self.assertRaises(ValueError):
            normalize_safety_decision(request, {'approved': False, 'status': 'Pass', 'reason': 'bad'})

    def test_actual_A_control_plan_success(self):
        decision, _ = load_decision_agent()
        result = run_a_plan(decision, {'goal_type': 'control_cycle'}, self.agent, self.monitor,
                            log_path=Path(self.temp.name) / 'main.jsonl')
        self.assertEqual(result['task_feedback']['status'], 'completed')
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertEqual(len(result['original_a_task']['subtasks']), 6)
        self.assertTrue(all(s['status'] == 'pending' for s in result['original_a_task']['subtasks']))

    def test_actual_A_control_plan_rejection_feedback(self):
        decision, _ = load_decision_agent()
        result = run_a_plan(decision, {'goal_type': 'control_cycle'}, self.agent, self.monitor,
                            explicit_action=[99, 0], log_path=Path(self.temp.name) / 'main.jsonl')
        self.assertEqual(result['task_feedback']['status'], 'blocked')
        self.assertEqual(self.adapter.step_calls, 0)
        by_name = {s['name']: s for s in result['task_feedback']['subtasks']}
        self.assertEqual(by_name['execute_approved_action']['status'], 'blocked')
        self.assertTrue(result['task_feedback']['failure_reason'])

    def test_actual_A_inspection_does_not_execute(self):
        decision, _ = load_decision_agent()
        result = run_a_plan(decision, '检查当前仿真状态', self.agent, self.monitor,
                            log_path=Path(self.temp.name) / 'main.jsonl')
        self.assertIsNone(result['action_request'])
        self.assertEqual(self.adapter.step_calls, 0)


if __name__ == '__main__':
    unittest.main()
