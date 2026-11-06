import unittest
from agents.scheduling_agent import SchedulingAgent
from algorithms.cnp_protocol import CNPError


def task(subtask="SUB-001"):
    return {
        "task_id": "TASK-001",
        "subtask_id": subtask,
        "task_type": "collect_observation",
        "priority": 3,
        "dependencies": [],
        "status": "pending",
    }


def bid(agent, cost=1.0, capability=0.9, load=0.2, subtask="SUB-001"):
    return {
        "task_id": "TASK-001",
        "subtask_id": subtask,
        "agent_id": agent,
        "estimated_cost": cost,
        "capability_match": capability,
        "load": load,
    }


class TestNov5FailureRecovery(unittest.TestCase):
    def setUp(self):
        self.scheduler = SchedulingAgent()
        self.scheduler.register_tasks([task()])

    def test_no_bids_returns_unassigned_for_timeout_or_unavailable(self):
        result = self.scheduler.schedule(task(), [])
        self.assertEqual(result["assignment_status"], "unassigned")
        self.assertIsNone(result["winner_id"])
        self.assertEqual(self.scheduler.get_task_status("TASK-001", "SUB-001"), "announced")

    def test_failed_winner_is_excluded_and_task_reassigned(self):
        first = self.scheduler.schedule(task(), [
            bid("ExecutionAgent-01", cost=0.5),
            bid("ExecutionAgent-02", cost=1.5),
        ])
        self.assertEqual(first["winner_id"], "ExecutionAgent-01")
        self.scheduler.update_feedback({
            "task_id": "TASK-001",
            "subtask_id": "SUB-001",
            "execution_status": "failed",
            "failure_reason": "execution_timeout",
        })
        retry = self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001",
            [bid("ExecutionAgent-01", cost=0.1), bid("ExecutionAgent-02", cost=1.5)]
        )
        self.assertEqual(retry["assignment_status"], "awarded")
        self.assertEqual(retry["winner_id"], "ExecutionAgent-02")
        self.assertEqual(self.scheduler.get_task_status("TASK-001", "SUB-001"), "awarded")

    def test_resource_conflict_excludes_agent_with_active_assignment(self):
        first = self.scheduler.schedule(task("SUB-001"), [bid("ExecutionAgent-01")])
        second_task = {
            "task_id": "TASK-001", "subtask_id": "SUB-002",
            "task_type": "evaluate_goal", "priority": 2,
            "dependencies": [], "status": "pending",
        }
        second_bids = [
            bid("ExecutionAgent-01", subtask="SUB-002"),
            bid("ExecutionAgent-02", subtask="SUB-002"),
        ]
        second = self.scheduler.schedule(second_task, second_bids)
        self.assertEqual(first["winner_id"], "ExecutionAgent-01")
        self.assertEqual(second["winner_id"], "ExecutionAgent-02")

    def test_completed_feedback_releases_agent_resource(self):
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "completed"
        })
        self.assertNotIn("ExecutionAgent-01", self.scheduler.active_agent_tasks)

    def test_failed_task_without_alternative_remains_failed(self):
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "resource_conflict"
        })
        result = self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-01")]
        )
        self.assertEqual(result["assignment_status"], "unassigned")
        self.assertEqual(self.scheduler.get_task_status("TASK-001", "SUB-001"), "failed")

    def test_retry_only_allowed_after_failure(self):
        with self.assertRaises(CNPError):
            self.scheduler.retry_failed_task("TASK-001", "SUB-001", [bid("ExecutionAgent-02")])


class TestNov6FeedbackAndAssignmentHistory(unittest.TestCase):
    def setUp(self):
        self.scheduler = SchedulingAgent()
        self.scheduler.register_tasks([task()])

    def test_failure_reason_categories_are_preserved(self):
        categories = [
            "safety_rejected",
            "execution_timeout",
            "agent_unavailable",
            "environment_error",
        ]
        for category in categories:
            # Separate scheduler per category so each feedback tests a fresh assignment.
            scheduler = SchedulingAgent()
            scheduler.register_tasks([task()])
            scheduler.schedule(task(), [bid("ExecutionAgent-01")])
            result = scheduler.update_feedback({
                "task_id": "TASK-001",
                "subtask_id": "SUB-001",
                "execution_status": "failed",
                "failure_reason": category,
            })
            self.assertEqual(result["failure_reason"], category)

    def test_safety_rejection_does_not_exclude_agent_by_default(self):
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        result = self.scheduler.update_feedback({
            "task_id": "TASK-001",
            "subtask_id": "SUB-001",
            "execution_status": "failed",
            "failure_reason": "safety_rejected",
        })
        self.assertFalse(result["exclude_failed_agent"])
        retry = self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-01")]
        )
        self.assertEqual(retry["winner_id"], "ExecutionAgent-01")

    def test_explicit_exclusion_flag_overrides_default(self):
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.scheduler.update_feedback({
            "task_id": "TASK-001",
            "subtask_id": "SUB-001",
            "execution_status": "failed",
            "failure_reason": "safety_rejected",
            "exclude_failed_agent": True,
        })
        retry = self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001",
            [bid("ExecutionAgent-01"), bid("ExecutionAgent-02")]
        )
        self.assertEqual(retry["winner_id"], "ExecutionAgent-02")

    def test_failed_assignment_history_is_preserved(self):
        first = self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "execution_timeout"
        })
        second = self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-02")]
        )
        history = self.scheduler.get_assignment_history("TASK-001", "SUB-001")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["winner_id"], first["winner_id"])
        self.assertEqual(history[1]["winner_id"], second["winner_id"])
        self.assertEqual(self.scheduler.get_assignment("TASK-001", "SUB-001")["winner_id"],
                         "ExecutionAgent-02")

    def test_old_agent_lock_released_after_failure_and_retry(self):
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "execution_timeout"
        })
        self.assertNotIn("ExecutionAgent-01", self.scheduler.active_agent_tasks)
        self.scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-02")]
        )
        self.assertNotIn("ExecutionAgent-01", self.scheduler.active_agent_tasks)
        self.assertEqual(self.scheduler.active_agent_tasks["ExecutionAgent-02"], "SUB-001")

    def test_no_bid_can_be_reannounced_without_terminal_failure(self):
        result = self.scheduler.schedule(task(), [])
        self.assertEqual(result["assignment_status"], "unassigned")
        self.assertEqual(self.scheduler.get_task_status("TASK-001", "SUB-001"), "announced")
        # schedule() accepts announced tasks, so A can call it again with fresh bids.
        second = self.scheduler.schedule(task(), [bid("ExecutionAgent-02")])
        self.assertEqual(second["assignment_status"], "awarded")
        self.assertEqual(second["winner_id"], "ExecutionAgent-02")

    def test_active_assignment_query(self):
        self.assertIsNone(self.scheduler.get_active_assignment("TASK-001", "SUB-001"))
        self.scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.assertEqual(
            self.scheduler.get_active_assignment("TASK-001", "SUB-001")["winner_id"],
            "ExecutionAgent-01"
        )


class TestNov6ReplanningIntegrationScenarios(unittest.TestCase):
    def test_agent_unavailable_reassigns_to_alternative(self):
        scheduler = SchedulingAgent()
        scheduler.register_tasks([task()])
        first = scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        self.assertEqual(first["winner_id"], "ExecutionAgent-01")
        scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "agent_unavailable",
        })
        second = scheduler.retry_failed_task(
            "TASK-001", "SUB-001",
            [bid("ExecutionAgent-01"), bid("ExecutionAgent-02")]
        )
        self.assertEqual(second["winner_id"], "ExecutionAgent-02")

    def test_environment_error_can_be_overridden_not_to_exclude(self):
        scheduler = SchedulingAgent()
        scheduler.register_tasks([task()])
        scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "environment_error",
            "exclude_failed_agent": False,
        })
        second = scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-01")]
        )
        self.assertEqual(second["winner_id"], "ExecutionAgent-01")

    def test_task_ids_are_preserved_across_retry_history(self):
        scheduler = SchedulingAgent()
        scheduler.register_tasks([task()])
        scheduler.schedule(task(), [bid("ExecutionAgent-01")])
        scheduler.update_feedback({
            "task_id": "TASK-001", "subtask_id": "SUB-001",
            "execution_status": "failed", "failure_reason": "execution_timeout",
        })
        scheduler.retry_failed_task(
            "TASK-001", "SUB-001", [bid("ExecutionAgent-02")]
        )
        history = scheduler.get_assignment_history("TASK-001", "SUB-001")
        self.assertTrue(all(x["task_id"] == "TASK-001" for x in history))
        self.assertTrue(all(x["subtask_id"] == "SUB-001" for x in history))


if __name__ == "__main__":
    unittest.main()
