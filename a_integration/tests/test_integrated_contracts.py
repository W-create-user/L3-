import unittest
from types import SimpleNamespace
from a_integration.decision_agent import DecisionAgent
from a_integration.agents.scheduling_agent import SchedulingAgent
from a_integration.adapters.a_to_c_adapter import convert_subtasks_for_c
from a_integration.adapters.b_to_d_adapter import make_d_safety_checker
from a_integration.adapters.assignment_adapter import assert_winner
from a_integration.adapters.feedback_adapter import execution_result_to_feedback, update_scheduler_from_result
from a_integration.main import execute_awarded_subtask


class FakeMonitor:
    demo_mode = False
    def __init__(self, decision="pass"):
        self.decision = decision
    def check_request(self, request, observation=None):
        return {"task_id": request["task_id"], "decision": self.decision,
                "rule_id": None if self.decision == "pass" else "R-02",
                "reason": "test", "risk_level": 0 if self.decision == "pass" else 3}


class FakeExecutor:
    def __init__(self):
        self.adapter = SimpleNamespace(step_count=0, step_calls=0)
        self.observation = [0.0] * 39
    def get_observation(self):
        return self.observation
    def execute_assignment(self, task, checker, *, action=None, action_policy=None, subtask=None):
        req = {"task_id": task["task_id"], "subtask_id": subtask["subtask_id"],
               "action": action if action is not None else [0.0, 0.0, 0.0, 0.0]}
        decision = checker(req, self.observation)
        if decision["decision"] != "pass":
            return {"task_id": task["task_id"], "subtask_id": subtask["subtask_id"],
                    "execution_status": "rejected", "success": False, "executed": False,
                    "failure_reason": decision["reason"], "safety_decision": decision}
        self.adapter.step_count += 1
        self.adapter.step_calls += 1
        return {"task_id": task["task_id"], "subtask_id": subtask["subtask_id"],
                "execution_status": "completed", "success": True, "executed": True,
                "failure_reason": None, "safety_decision": decision}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        obj = DecisionAgent().create_task({"task_type":"control_cycle","description":"test","priority":3})
        self.parent = DecisionAgent.task_to_dict(obj)
        self.c_tasks = convert_subtasks_for_c(self.parent)
        self.scheduler = SchedulingAgent()
        self.scheduler.register_tasks(self.c_tasks)

    def test_real_d_adapter_contract_pass(self):
        checker = make_d_safety_checker(FakeMonitor("pass"))
        out = checker({"task_id":"T1","action":[0,0,0,0]}, [0]*39)
        self.assertEqual(out["decision"], "pass")
        self.assertFalse(out["demo_mode"])

    def test_reject_feedback_maps_to_failed(self):
        fb = execution_result_to_feedback({"task_id":"T","subtask_id":"S",
             "execution_status":"rejected","failure_reason":"R-02",
             "safety_decision":{"rule_id":"R-02"}})
        self.assertEqual(fb["execution_status"], "failed")
        self.assertIn("safety_rejected", fb["failure_reason"])
        self.assertFalse(fb["exclude_failed_agent"])

    def test_default_executor_identity(self):
        from a_integration.config import EXECUTOR_ID
        self.assertEqual(EXECUTOR_ID, "ExecutionAgent-01")

    def test_winner_mismatch_blocks(self):
        with self.assertRaises(PermissionError):
            assert_winner({"assignment_status":"awarded","winner_id":"other"},"ExecutionAgent-01")

    def test_readonly_subtask_does_not_step(self):
        task = self.c_tasks[0]
        assignment = self.scheduler.schedule(task, [{
            "task_id":task["task_id"],"subtask_id":task["subtask_id"],
            "agent_id":"ExecutionAgent-01","estimated_cost":1,
            "capability_match":0.9,"load":0.2}])
        executor = FakeExecutor()
        result = execute_awarded_subtask(parent_task=self.parent,c_task=task,
            assignment=assignment,scheduler=self.scheduler,executor=executor,
            monitor=FakeMonitor(),executor_id="ExecutionAgent-01")
        self.assertEqual(result["execution_result"]["execution_status"],"completed")
        self.assertEqual(executor.adapter.step_calls,0)
        self.assertEqual(self.scheduler.get_task_status(task["task_id"],task["subtask_id"]),"completed")

    def test_control_subtask_pass_and_feedback(self):
        # Unlock the control subtask only by recording completion of its predecessors.
        for task in self.c_tasks[:4]:
            self.scheduler.update_feedback({"task_id":task["task_id"],
                "subtask_id":task["subtask_id"],"execution_status":"completed"})
        task = self.c_tasks[4]
        assignment = self.scheduler.schedule(task, [{
            "task_id":task["task_id"],"subtask_id":task["subtask_id"],
            "agent_id":"ExecutionAgent-01","estimated_cost":1,
            "capability_match":0.9,"load":0.2}])
        executor = FakeExecutor()
        result = execute_awarded_subtask(parent_task=self.parent,c_task=task,
            assignment=assignment,scheduler=self.scheduler,executor=executor,
            monitor=FakeMonitor("pass"),executor_id="ExecutionAgent-01")
        self.assertTrue(result["execution_result"]["executed"])
        self.assertEqual(executor.adapter.step_calls,1)
        self.assertEqual(self.scheduler.get_task_status(task["task_id"],task["subtask_id"]),"completed")

def test_control_subtask_reject_does_not_step(self):
    for task in self.c_tasks[:4]:
        self.scheduler.update_feedback({"task_id":task["task_id"],
            "subtask_id":task["subtask_id"],"execution_status":"completed"})
    task = self.c_tasks[4]
    assignment = self.scheduler.schedule(task, [{
        "task_id":task["task_id"],"subtask_id":task["subtask_id"],
        "agent_id":"ExecutionAgent-01","estimated_cost":1,
        "capability_match":0.9,"load":0.2}])
    executor = FakeExecutor()
    result = execute_awarded_subtask(parent_task=self.parent,c_task=task,
        assignment=assignment,scheduler=self.scheduler,executor=executor,
        monitor=FakeMonitor("reject"),executor_id="ExecutionAgent-01")
    self.assertFalse(result["execution_result"]["executed"])
    self.assertEqual(executor.adapter.step_calls,0)
    self.assertEqual(self.scheduler.get_task_status(task["task_id"],task["subtask_id"]),"failed")
    self.assertFalse(self.scheduler.get_assignment_history(task["task_id"],task["subtask_id"])[0]["winner_id"] == "other")

    def test_d_demo_mode_refused(self):
        monitor = FakeMonitor()
        monitor.demo_mode = True
        with self.assertRaises(RuntimeError):
            make_d_safety_checker(monitor)


if __name__ == "__main__":
    unittest.main()
