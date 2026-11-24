"""A07 local fault matrix with real A/B/C/D classes and a counting adapter."""
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from l3_system.agents.monitoring_agent import MonitoringAgent
from l3_system.execution import ExecutionAgent
from l3_system.main import configured_bids
from l3_system.planning.bid_routing import BidRoute
from l3_system.runtime import MultiAgentSystem
from l3_system.tests.test_live_orchestration_contracts import FakeAdapter
from l3_system.ui.app import create_app


class LocalFaultMatrix(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.adapter = FakeAdapter()
        self.executor = ExecutionAgent(self.adapter, log_path=root / "B.jsonl")
        monitor = MonitoringAgent(self.adapter.action_space, self.adapter.observation_space)
        self.system = MultiAgentSystem(executor=self.executor, monitor=monitor,
                                       event_path=root / "A.jsonl")
        self.client = create_app(self.system, environment="AtropineEnvGym", separate_confirmation=True).test_client()

    def submit(self, operation, **updates):
        page = self.client.get("/").get_data(as_text=True)
        form = dict(re.findall(r'name="(token|revision)" value="([^"]+)"', page))
        return self.client.post("/" + operation,
                                data={**form, "operator": "test", "reason": "fixture", **updates})

    def test_pause_and_confirm_resume_are_audited_without_step(self):
        previous = self.system.run_cycle()
        self.assertEqual(self.submit("pause").status_code, 303)
        self.assertEqual(self.submit("cycle").status_code, 409)
        self.assertEqual(self.submit("resume").status_code, 409)
        self.assertEqual(self.submit("confirm").status_code, 303)
        self.assertEqual(self.submit("resume").status_code, 303)
        self.assertEqual(self.adapter.step_calls, 1)
        events = [json.loads(line) for line in self.system.event_path.read_text().splitlines()]
        expected = ["operator_requested_pause", "operator_confirmed_takeover", "operator_resumed_console"]
        operations = [event for event in events if event.get("reason") in expected]
        self.assertEqual([event["reason"] for event in operations], expected)
        self.assertTrue(all(event["task_id"] == previous["summary"]["task_id"] for event in operations))

    def test_resume_with_stale_confirmation_invalid_observation_or_lock(self):
        self.submit("pause")
        self.submit("confirm")
        self.adapter.state_version += 1
        self.assertEqual(self.submit("resume").status_code, 409)
        self.submit("confirm")
        self.adapter._obs[0] = float("nan")
        self.assertEqual(self.submit("resume").status_code, 409)
        self.adapter._obs[0] = 0
        active = self.system.scheduler.active_agent_tasks
        active["worker"] = "unresolved-assignment"
        self.assertEqual(self.submit("resume", reset="yes").status_code, 409)
        self.assertEqual(active["worker"], "unresolved-assignment")
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertEqual(self.system.state, "takeover_requested")

    def test_goal_revision_requires_pause_and_invalidates_takeover_confirmation(self):
        config = Path(__file__).resolve().parents[2] / "configs/goals"
        goal = json.loads((config / "atropine_goal_regression.json").read_text())
        registry = json.loads((config / "atropine_model_registry.json").read_text())
        self.client = create_app(self.system, environment="AtropineEnvGym",
                                 goal=goal, registry=registry, separate_confirmation=True).test_client()
        self.assertEqual(self.submit("revise", lower="0.01", upper="0.03").status_code, 409)
        self.submit("pause")
        self.submit("confirm")
        self.assertEqual(self.submit("revise", lower="2", upper="1").status_code, 400)
        self.assertEqual(self.submit("revise", lower="0.01", upper="0.03").status_code, 303)
        self.assertEqual(self.submit("resume").status_code, 409)
        self.submit("confirm")
        self.assertEqual(self.submit("resume").status_code, 303)
        self.assertEqual(self.submit("cycle").status_code, 303)
        self.assertEqual(self.adapter.step_calls, 2)
        self.assertIn("succeeded", self.client.get("/").get_data(as_text=True))

    def test_missing_prerequisite_prevents_offer_or_step(self):
        result = self.system.run_cycle(goal={
            "task_type": "control_cycle", "dependencies": ["missing-parent"],
        })
        self.assertEqual(result["summary"]["failure_reason"], "dependency_blocked")
        self.assertEqual(self.adapter.step_calls, 0)

    def test_worker_disconnect_cancels_prepared_request(self):
        def bids(task):
            if task["task_type"] == "execute_approved_action":
                raise ConnectionError("worker exited before the award")
            return configured_bids(task)
        self.system.bid_provider = bids
        result = self.system.run_cycle()
        self.assertEqual(result["summary"]["failure_reason"], "no_resource")
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertFalse(self.system.scheduler.active_agent_tasks)
        self.assertFalse(self.executor._pending)
        self.assertEqual(self.system.state, "takeover_requested")

    def test_cancellation_keeps_lock_when_release_is_unconfirmed(self):
        original = self.system.scheduler.update_feedback
        def fail_on_intent(feedback):
            if feedback["subtask_id"] == list(self.system.scheduler.tasks)[2]:
                raise RuntimeError("failure confirming release")
            return original(feedback)
        with patch.object(self.system.scheduler, "update_feedback", side_effect=fail_on_intent):
            with self.assertRaises(RuntimeError):
                self.system.run_cycle()
        self.assertFalse(self.executor._pending)
        self.assertTrue(self.system.scheduler.active_agent_tasks)
        self.assertEqual(self.adapter.step_calls, 0)
        self.submit("confirm")
        self.assertEqual(self.submit("resume", reset="yes").status_code, 409)

    def test_unknown_dispatch_does_not_retry_even_without_step_counter(self):
        with patch.object(self.executor, "execute_action", side_effect=TimeoutError("reply lost")) as dispatch:
            result = self.system.run_cycle()
        dispatch.assert_called_once()
        self.assertEqual(result["summary"]["failure_reason"], "execution_result_unknown")
        self.assertFalse(self.executor._pending)
        self.submit("confirm")
        self.assertEqual(self.submit("resume").status_code, 409)
        self.assertEqual(self.adapter.step_calls, 0)

    def test_old_bid_reply_is_not_used_for_the_next_subtask(self):
        first = None
        def exchange(request, *, timeout_seconds):
            nonlocal first
            self.assertEqual(timeout_seconds, 2)
            if first is None:
                first = {**request, "bids": configured_bids(request["task"])}
            return first
        self.system.bid_provider = BidRoute(exchange, environment="AtropineEnvGym")
        result = self.system.run_cycle()
        self.assertEqual(result["summary"]["completed_subtasks"], 1)
        self.assertEqual(self.adapter.step_calls, 0)
        self.assertEqual(self.system.state, "takeover_requested")
