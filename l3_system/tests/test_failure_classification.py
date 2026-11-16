import unittest
from unittest.mock import Mock
from l3_system.planning.replanning import FailureType, ReplanningController, classify_failure


class FailureClassificationTests(unittest.TestCase):
    def result(self, reason, **facts):
        return dict(task_id='T', subtask_id='S', execution_status='failed', failure_reason=reason, **facts)

    def test_six_failure_categories(self):
        cases = [
            (self.result('safety_rejected', executed=False, step_called=False), FailureType.CANDIDATE_REJECTED),
            (self.result('no_resource'), FailureType.NO_RESOURCE),
            (self.result('message_timeout', executed=False, step_called=False), FailureType.MESSAGE_TIMEOUT),
            (self.result('execution_timeout'), FailureType.EXECUTION_UNKNOWN),
            (self.result('environment_error'), FailureType.ENVIRONMENT_ERROR),
            (self.result('error mentioning agent'), FailureType.UNKNOWN_STATE)]
        for value, expected in cases:
            with self.subTest(expected=expected): self.assertEqual(classify_failure(value), expected)

    def test_timeout_without_nonexecution_proof_never_calls_scheduler(self):
        controller, scheduler = ReplanningController(), Mock()
        for facts in ({}, {'executed':False}, {'executed':False,'step_called':True},
                      {'executed':True,'step_called':False}):
            outcome = controller.retry_assignment(scheduler, self.result('execution_timeout', **facts), [])
            self.assertEqual(outcome['assignment_status'], 'not_retried')
        scheduler.retry_failed_task.assert_not_called()

    def test_release_confirmation_and_actual_lock_state_required(self):
        controller, scheduler = ReplanningController(), Mock()
        facts = dict(executed=False, step_called=False)
        outcome = controller.retry_assignment(scheduler, self.result('no_resource', **facts), [])
        self.assertEqual(outcome['assignment_status'], 'not_retried')
        facts['release_confirmed'] = True
        scheduler.active_agent_tasks = {'busy':'S'}
        self.assertEqual(controller.retry_assignment(scheduler, self.result('no_resource', **facts), [])['reason'], 'release_not_confirmed')
        scheduler.retry_failed_task.assert_not_called()

    def test_retry_budget_is_bounded(self):
        controller, scheduler = ReplanningController(max_environment_retries=1), Mock()
        scheduler.active_agent_tasks = {}
        scheduler.retry_failed_task.return_value = {'assignment_status':'awarded'}
        failure = self.result('no_resource', executed=False, step_called=False, release_confirmed=True)
        self.assertEqual(controller.retry_assignment(scheduler, failure, [])['assignment_status'], 'awarded')
        self.assertEqual(controller.retry_assignment(scheduler, failure, [])['assignment_status'], 'not_retried')
        scheduler.retry_failed_task.assert_called_once()

    def test_contradictory_safety_rejection_cannot_replay(self):
        failure = self.result('safety_rejected', executed=True, step_called=True)
        self.assertEqual(ReplanningController().classify(failure)['action'], 'reconcile_then_takeover')

    def test_invalid_budgets_and_unknown_inputs(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError): ReplanningController(max_environment_retries=value)
        self.assertFalse(ReplanningController().classify(None)['retryable'])
