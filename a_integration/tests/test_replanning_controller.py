import unittest
from a_integration.agents.scheduling_agent import SchedulingAgent
from a_integration.planning.replanning import ReplanningController


class ReplanningControllerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = SchedulingAgent()
        self.task = {"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
                     "task_type": "execute_approved_action", "priority": 3,
                     "dependencies": [], "status": "pending", "constraints": {}}
        self.scheduler.register_tasks([self.task])
        self.initial_bids = [{"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
                              "agent_id": "ExecutionAgent-01", "estimated_cost": 1,
                              "capability_match": .95, "load": .1}]
        self.scheduler.schedule(self.task, self.initial_bids)

    def test_safety_rejection_requests_action_regeneration_not_agent_exclusion(self):
        self.scheduler.update_feedback({"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
            "execution_status": "failed", "failure_reason": "safety_rejected",
            "exclude_failed_agent": False})
        controller = ReplanningController()
        result = {"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
                  "execution_status": "rejected", "safety_decision": {"rule_id": "R-02"}}
        outcome = controller.retry_assignment(self.scheduler, result, self.initial_bids)
        self.assertEqual(outcome["assignment_status"], "action_regeneration_required")
        self.assertEqual(controller.action_regenerations, {})

    def test_execution_timeout_can_reassign_to_alternative(self):
        self.scheduler.update_feedback({"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
            "execution_status": "failed", "failure_reason": "execution_timeout",
            "exclude_failed_agent": True})
        controller = ReplanningController()
        result = {"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01",
                  "execution_status": "failed", "failure_reason": "execution_timeout"}
        bids = [
            {"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01", "agent_id": "ExecutionAgent-01",
             "estimated_cost": .1, "capability_match": .99, "load": .1},
            {"task_id": "P-REPLAN", "subtask_id": "P-REPLAN-01", "agent_id": "ExecutionAgent-02",
             "estimated_cost": 1, "capability_match": .8, "load": .2},
        ]
        outcome = controller.retry_assignment(self.scheduler, result, bids)
        self.assertEqual(outcome["assignment_status"], "awarded")
        self.assertEqual(outcome["winner_id"], "ExecutionAgent-02")


if __name__ == '__main__':
    unittest.main()
