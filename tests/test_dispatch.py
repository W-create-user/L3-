import unittest
from runtime.dispatch import Worker, Dispatcher
from runtime.supervision import Supervision


def task(key='s', dependencies=None):
    return {'task_id': 'parent', 'subtask_id': key, 'task_type': 'control',
            'dependencies': dependencies or []}


class DispatchTests(unittest.TestCase):
    def workers(self):
        return [Worker('busy', {'execute'}, cost=10, load=.9),
                Worker('available', {'execute'}, cost=1, load=.1)]

    def test_cnp_selects_available_worker(self):
        result = Dispatcher(self.workers()).execute([task()], {'s': 'execute'}, lambda *args: True)
        self.assertTrue(result['success'])
        self.assertEqual(result['events'][0]['assignment']['winner_id'], 'available')

    def test_retry_excludes_failed_worker(self):
        d = Dispatcher(self.workers())
        result = d.execute([task()], {'s': 'execute'}, lambda worker, task: worker == 'busy')
        self.assertTrue(result['success'])
        self.assertEqual([e['assignment']['winner_id'] for e in result['events']], ['available', 'busy'])
        self.assertFalse(d.scheduler.active_agent_tasks)

    def test_dependency_order(self):
        result = Dispatcher(self.workers()).execute([task('b', ['a']), task('a')],
                    {'a': 'execute', 'b': 'execute'}, lambda *args: True)
        self.assertEqual([e['feedback']['subtask_id'] for e in result['events']], ['a', 'b'])

    def test_cycle_stops_without_execution(self):
        result = Dispatcher(self.workers()).execute([task('a', ['b']), task('b', ['a'])],
                    {'a': 'execute', 'b': 'execute'}, lambda *args: self.fail('Must not execute'))
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'unresolved_dependencies')

    def test_missing_capability(self):
        self.assertFalse(Dispatcher(self.workers()).execute([task()], {'s': 'unknown'},
                                                          lambda *args: True)['success'])

    def test_takeover_latches_until_named_confirmation(self):
        state = Supervision()
        state.request_takeover('safety_rejected')
        with self.assertRaises(RuntimeError):
            state.require_autonomous()
        with self.assertRaises(ValueError):
            state.confirm_recovery('', 'checked')
        state.confirm_recovery('operator-1', 'checked in simulator')
        state.require_autonomous()
