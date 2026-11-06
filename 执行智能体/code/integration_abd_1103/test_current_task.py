import copy
import json
from pathlib import Path
import tempfile
import unittest

from .a_integration import run_task_dict
from .execution_agent import ExecutionAgent
from .reference_monitor import ReferenceMonitoringAgent
from .task_contract import normalize_task
from .test_contract import CountingAdapter


class CurrentTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.adapter = CountingAdapter()
        self.executor = ExecutionAgent(self.adapter, Path(self.temp.name) / 'B.jsonl')
        self.executor.reset()
        self.monitor = ReferenceMonitoringAgent(self.adapter.action_space)
        self.task = json.loads((Path(__file__).parent / 'examples/a_task_current.json').read_text(encoding='utf-8'))

    def run_task(self, task, **kwargs):
        return run_task_dict(task, self.executor, self.monitor,
                             log_path=Path(self.temp.name) / 'nested/main.jsonl', **kwargs)

    def test_raw_A_task_without_action_uses_B_policy_and_preserves_input(self):
        original = copy.deepcopy(self.task)
        request = self.executor.prepare_action(self.task)
        self.assertEqual(self.task, original)
        self.assertEqual(request['task_id'], 'TASK-5b95693d')
        self.assertEqual(request['task_name'], self.task['description'])
        self.assertEqual(request['subtask_id'], 'SUB-56947c01')
        self.assertEqual(request['action_source'], 'B_local_space_midpoint_baseline')
        self.assertEqual(self.adapter.step_calls, 0)

    def test_three_field_mapping_alone_is_enough(self):
        mapped = normalize_task(self.task)
        self.assertEqual(set(mapped), {'task_id', 'task_name', 'parameters'})
        self.assertNotIn('action', mapped['parameters'])
        request = self.executor.prepare_action(mapped)
        self.assertEqual(request['subtask_id'], 'SUB-56947c01')
        self.assertEqual(request['task_context']['parameters']['subtasks'], self.task['subtasks'])

    def test_metadata_is_not_discarded(self):
        request = self.executor.prepare_action(self.task)
        metadata = request['task_context']['parameters']
        for field in ['task_type','priority','preconditions','dependencies','constraints','subtasks','completion_criteria','status','source']:
            self.assertEqual(metadata[field], self.task[field])

    def test_raw_and_mapped_produce_same_candidate(self):
        raw = self.executor.prepare_action(self.task)
        mapped = self.executor.prepare_action(normalize_task(self.task))
        self.assertEqual(raw['action'], mapped['action'])
        self.assertNotEqual(raw['request_id'], mapped['request_id'])
        self.assertEqual(self.adapter.step_calls, 0)

    def test_new_preconditions_and_completion_criteria_work(self):
        result = self.run_task(self.task)
        self.assertEqual(result['task_status'], 'completed')
        feedback = result['task_feedback']
        self.assertEqual(feedback['satisfied_completion_criteria'], ['all_subtasks_completed'])
        self.assertTrue(all(s['status'] == 'completed' for s in feedback['subtasks']))
        self.assertEqual(self.adapter.step_calls, 1)

    def test_mapped_plan_also_works(self):
        result = self.run_task(normalize_task(self.task))
        self.assertEqual(result['task_status'], 'completed')
        self.assertEqual(result['task_feedback']['parameters']['status'], 'completed')

    def test_rejected_plan_does_not_satisfy_parent_completion(self):
        result = self.run_task(self.task, explicit_action=[99, 0])
        self.assertEqual(result['task_status'], 'blocked')
        self.assertEqual(result['task_feedback']['satisfied_completion_criteria'], [])
        self.assertEqual(self.adapter.step_calls, 0)

    def test_nested_subtask_selection_is_supported(self):
        task = normalize_task(self.task)
        request = self.executor.prepare_action(task, subtask=task['parameters']['subtasks'][4])
        self.assertEqual(request['subtask_id'], 'SUB-56947c01')

    def test_conflicting_metadata_is_refused(self):
        task = copy.deepcopy(self.task)
        task['parameters'] = {'constraints': {'unexpected': 1}}
        with self.assertRaises(ValueError):
            self.executor.prepare_action(task)

    def test_conflicting_task_type_is_refused(self):
        task = copy.deepcopy(self.task)
        task['goal_type'] = 'inspect_state'
        with self.assertRaises(ValueError):
            self.executor.prepare_action(task)

    def test_invalid_current_observation_cannot_generate_candidate(self):
        self.adapter.get_observation = lambda: [float('nan')]
        with self.assertRaises(ValueError):
            self.executor.prepare_action(self.task)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_unknown_type_does_not_trigger_an_implicit_policy(self):
        with self.assertRaises(ValueError):
            self.executor.prepare_action({'task_id': 'unknown', 'task_type': 'unknown'})

    def test_unresolved_parent_dependencies_cannot_execute(self):
        task = copy.deepcopy(self.task)
        task['dependencies'] = ['TASK-prerequisite']
        with self.assertRaises(RuntimeError):
            self.run_task(task)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_unknown_completion_criteria_are_not_marked_complete(self):
        task = copy.deepcopy(self.task)
        task['subtasks'][-1]['completion_criteria'] = ['production_quality_verified']
        with self.assertRaises(ValueError):
            self.run_task(task)
        self.assertEqual(self.adapter.step_calls, 0)


if __name__ == '__main__':
    unittest.main()
