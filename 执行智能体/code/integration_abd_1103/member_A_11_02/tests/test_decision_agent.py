import unittest

from agents.decision_agent import DecisionAgent


class DecisionAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = DecisionAgent()

    def test_control_cycle_has_six_ordered_subtasks(self):
        task = self.agent.create_task({
            "goal_type": "control_cycle",
            "description": "完成一次仿真控制任务",
            "priority": 4,
            "constraints": {"environment": "AtropineEnv"},
        })
        self.assertEqual(len(task.subtasks), 6)
        self.assertEqual([s.order for s in task.subtasks], [1, 2, 3, 4, 5, 6])
        self.assertEqual(task.subtasks[0].dependencies, [])
        for previous, current in zip(task.subtasks, task.subtasks[1:]):
            self.assertEqual(current.dependencies, [previous.subtask_id])

    def test_monitoring_precedes_execution(self):
        task = self.agent.create_task({"goal_type": "control_cycle"})
        names = [s.name for s in task.subtasks]
        self.assertLess(names.index("safety_review"),
                        names.index("execute_approved_action"))

    def test_inspect_state_template(self):
        task = self.agent.create_task("检查当前仿真状态")
        self.assertEqual(task.goal_type, "inspect_state")
        self.assertEqual(len(task.subtasks), 3)

    def test_priority_validation(self):
        for value in (0, 6, True, "4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.agent.create_task({
                        "goal_type": "control_cycle",
                        "priority": value,
                    })

    def test_unsupported_goal_rejected(self):
        with self.assertRaises(ValueError):
            self.agent.create_task("执行未经定义的任务")

    def test_json_serializable_output(self):
        task = self.agent.create_task({"goal_type": "control_cycle"})
        payload = self.agent.task_to_dict(task)
        import json
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("subtasks", encoded)
        self.assertIn("dependencies", encoded)

    def test_all_dependencies_reference_existing_subtasks(self):
        task = self.agent.create_task({"goal_type": "control_cycle"})
        ids = {s.subtask_id for s in task.subtasks}
        for subtask in task.subtasks:
            self.assertTrue(set(subtask.dependencies).issubset(ids))


if __name__ == "__main__":
    unittest.main()
