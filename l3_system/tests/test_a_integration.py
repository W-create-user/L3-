import unittest
from l3_system.decision_agent import DecisionAgent
from l3_system.adapters.a_to_c_adapter import convert_subtasks_for_c, enrich_assignment
from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.adapters.runtime_adapters import make_c_feedback_callback

class AIntegrationTests(unittest.TestCase):
    def setUp(self):
        agent = DecisionAgent()
        task = agent.create_task({"task_type":"control_cycle", "description":"test", "priority":4,
                                  "constraints":{"environment":"AtropineEnv"}})
        self.decision = agent.task_to_dict(task)
        self.tasks = convert_subtasks_for_c(self.decision)
        self.scheduler = SchedulingAgent()
        self.scheduler.register_tasks(self.tasks)

    def test_six_tasks_registered(self):
        self.assertEqual(len(self.tasks), 6)
        self.assertEqual(len(self.scheduler.tasks), 6)

    def test_unique_subtask_ids(self):
        ids = [x["subtask_id"] for x in self.tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_first_ready_only(self):
        ready = self.scheduler.ready_tasks()
        self.assertEqual([x["task_type"] for x in ready], ["collect_observation"])

    def test_dependencies_reference_known_ids(self):
        ids = {x["subtask_id"] for x in self.tasks}
        for task in self.tasks:
            self.assertTrue(set(task["dependencies"]).issubset(ids))

    def test_assignment_enrichment(self):
        t = self.tasks[0]
        a = {"task_id":t["task_id"], "subtask_id":t["subtask_id"], "winner_id":"ExecutionAgent-01",
             "score":0.8, "assignment_status":"awarded"}
        self.assertEqual(enrich_assignment(a, self.tasks)["task_type"], "collect_observation")

    def test_feedback_completed_unlocks_next(self):
        t = self.tasks[0]
        callback = make_c_feedback_callback(self.scheduler)
        callback({"task_id":t["task_id"], "subtask_id":t["subtask_id"],
                  "execution_status":"completed", "failure_reason":None})
        self.assertEqual(self.scheduler.ready_tasks()[0]["task_type"], "evaluate_goal")

    def test_rejected_normalized_to_failed(self):
        t = self.tasks[0]
        callback = make_c_feedback_callback(self.scheduler)
        callback({"task_id":t["task_id"], "subtask_id":t["subtask_id"],
                  "execution_status":"rejected", "failure_reason":"R-02"})
        self.assertEqual(self.scheduler.get_task_status(t["task_id"], t["subtask_id"]), "failed")

if __name__ == "__main__":
    unittest.main()
