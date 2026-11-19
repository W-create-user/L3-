import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import ExecutionAgent
from l3_system.planning.goal_runtime import run_numeric_goal
from l3_system.runtime import MultiAgentSystem
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.ui.app import create_app


class NumericGoalRuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.adapter = FakeAdapter()
        executor = ExecutionAgent(self.adapter, log_path=Path(directory.name) / "B.jsonl")
        monitor = MonitoringAgent(self.adapter.action_space, self.adapter.observation_space)
        self.system = MultiAgentSystem(executor=executor, monitor=monitor)
        root = Path(__file__).resolve().parents[2] / "configs/goals"
        self.registry = json.loads((root / "atropine_model_registry.json").read_text())
        self.goal = json.loads((root / "atropine_goal_regression.json").read_text())
        self.goal["target_range"] = {"lower": 0.01, "upper": 0.04}

    def run_goal(self, **kwargs):
        return run_numeric_goal(self.system, self.goal, self.registry, environment="AtropineEnvGym", **kwargs)

    def test_adjust_then_hold_and_report_through_existing_chain(self):
        result = self.run_goal(policy=lambda *_: [0] * 4)
        self.assertEqual(result["summary"]["status"], "succeeded")
        self.assertEqual([plan["method"] for plan in result["plans"]],
                         ["adjust_target", "hold_target", "report_achieved"])
        self.assertEqual([cycle["summary"]["step_delta"] for cycle in result["cycles"]], [1, 1, 0])
        self.assertEqual(result["plans"][1]["parent_plan_id"], result["plans"][0]["plan_id"])
        self.assertTrue(all(cycle["summary"]["all_C_locks_released"] for cycle in result["cycles"]))

    def test_prepared_candidate_outside_goal_range_does_not_step(self):
        result = self.run_goal(policy=lambda *_: [0.5] * 4)
        self.assertEqual(result["summary"]["status"], "failed")
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(self.system.executor._pending)

    def test_deadline_rechecked_after_policy_before_step(self):
        # A candidate becomes available after the deadline. D still cannot cause a step.
        clock = [0]
        def slow_policy(*_):
            clock[0] = 10
            return [0] * 4
        with patch("l3_system.planning.goal_runtime.time.monotonic", side_effect=lambda: clock[0]):
            result = self.run_goal(timeout_seconds=1, policy=slow_policy)
        self.assertEqual(result["summary"]["reason"], "goal_deadline_exceeded")
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(self.system.executor._pending)

    def test_stale_feedback_after_a_real_call_is_not_replayed(self):
        execute = self.system.executor.execute_action
        def wrong_request(*args):
            result = execute(*args)
            return {**result, "request_id": "stale-request"}
        with patch.object(self.system.executor, "execute_action", side_effect=wrong_request):
            result = self.run_goal(policy=lambda *_: [0] * 4)
        self.assertEqual(result["summary"]["status"], "failed")
        self.assertEqual(self.adapter.step_calls, 1)
        self.assertEqual(self.system.state, "takeover_requested")
        self.assertFalse(self.system.scheduler.active_agent_tasks)

    def test_web_operations_require_current_form_and_explicit_resume(self):
        import re
        client = create_app(self.system, environment="AtropineEnvGym").test_client()
        page = client.get("/").get_data(as_text=True)
        token = re.search(r'name="token" value="([^"]+)"', page)[1]
        self.assertEqual(client.post("/cycle", data={"revision": "0"}).status_code, 403)
        form = {"token": token, "revision": "0"}
        self.assertEqual(client.post("/cycle", data=form).status_code, 303)
        self.assertEqual(client.post("/cycle", data=form).status_code, 409)
        self.assertEqual(self.adapter.step_calls, 1)

        self.system._transition("takeover_requested", "test")
        self.assertEqual(client.post("/resume", data={**form, "revision": "1"}).status_code, 400)
        self.assertEqual(client.post("/resume", data={**form, "revision": "2",
                                                    "operator": "test", "reason": "checked"}).status_code, 303)
        self.assertEqual(self.system.state, "autonomous")
        self.assertEqual(self.adapter.step_calls, 1)
