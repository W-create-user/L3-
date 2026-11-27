import unittest

from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.algorithms.cnp_protocol import CNPError
from l3_system.scheduling.leased_scheduler import LeasedScheduler
from l3_system.scheduling.worker_registry import Worker, WorkerRegistry


def task(subtask="j1"):
    return {"task_id": "fixture", "subtask_id": subtask, "task_type": "regular"}


def bid(agent, subtask="j1", *, cost=1, load=0):
    return {"task_id": "fixture", "subtask_id": subtask, "agent_id": agent,
            "estimated_cost": cost, "capability_match": 1, "load": load}


class TestDynamicBidding(unittest.TestCase):
    def test_capability_environment_heartbeat_and_load(self):
        registry = WorkerRegistry(3)
        for name, env, capability in (("r1", "sim", "regular"), ("r2", "sim", "regular"),
                                      ("r3", "sim", "other"), ("r4", "other", "regular")):
            registry.register(Worker(name, frozenset({capability}), frozenset({env}), 1, 0))
        registry.update(registry.workers["r1"], heartbeat=0, load=1)
        generated = registry.bids(task(), now=1, environment="sim", base_cost=2, occupied=())
        self.assertEqual([b["agent_id"] for b in generated], ["r1", "r2"])
        ranking = LeasedScheduler().schedule(task(), generated, now=1, lease_duration=3)
        self.assertEqual(ranking["winner_id"], "r2")
        self.assertEqual(registry.bids(task(), now=3, environment="sim", base_cost=2,
                                       occupied=()), [])
        with self.assertRaises(CNPError):
            registry.register(registry.workers["r1"])

    def test_no_bids_and_invalid_candidates(self):
        scheduler = LeasedScheduler()
        self.assertEqual(scheduler.schedule(task(), [], now=0, lease_duration=2)["reason"],
                         "no_valid_bids")
        for invalid in ([bid("r1"), bid("r1")], [bid("r1", "old")],
                        [bid("r1", cost=float("nan"))]):
            with self.subTest(invalid=invalid), self.assertRaises(CNPError):
                scheduler.schedule(task(), invalid, now=1, lease_duration=2)


class TestLease(unittest.TestCase):
    def setUp(self):
        self.scheduler = LeasedScheduler(assignment_namespace="test")
        record = self.scheduler.schedule(task(), [bid("r1")], now=0, lease_duration=2)
        self.first = self.scheduler.assignment_for_execution(record, now=0)

    def test_timeout_requires_reconciliation_before_retry(self):
        scheduler = self.scheduler
        with self.assertRaisesRegex(CNPError, "expired"):
            scheduler.assignment_for_execution(self.first, now=2)
        self.assertEqual(scheduler.active_agent_tasks, {"r1": "j1"})
        receipt = {**self.first, "execution_status": "failed",
                   "outcome": "not_executed", "failure_reason": "confirmed_cancel"}
        with self.assertRaisesRegex(CNPError, "reconciliation"):
            scheduler.update_feedback(receipt, now=2)
        with self.assertRaisesRegex(CNPError, "proof"):
            scheduler.retry_failed_task("fixture", "j1", [bid("r2")], now=2, lease_duration=2)
        scheduler.update_feedback(receipt, now=2, reconciled=True)
        second = scheduler.retry_failed_task("fixture", "j1", [bid("r2")], now=2, lease_duration=2)
        self.assertEqual(second["attempt"], 2)
        with self.assertRaisesRegex(CNPError, "current assignment"):
            scheduler.update_feedback(receipt, now=2, reconciled=True)
        self.assertEqual(scheduler.active_agent_tasks, {"r2": "j1"})

    def test_identical_receipt_is_idempotent_and_wrong_identity_rejected(self):
        scheduler = self.scheduler
        receipt = {**self.first, "execution_status": "completed", "outcome": "executed"}
        result = scheduler.update_feedback(receipt, now=1)
        scheduler.schedule(task("j2"), [bid("r1", "j2")], now=1, lease_duration=2)
        self.assertEqual(scheduler.update_feedback(receipt, now=1), result)
        with self.assertRaisesRegex(CNPError, "current assignment"):
            scheduler.update_feedback({**receipt, "task_id": "wrong"}, now=1)
        self.assertEqual(scheduler.active_agent_tasks, {"r1": "j2"})

    def test_executed_failure_does_not_allow_retry(self):
        self.scheduler.update_feedback({**self.first, "execution_status": "failed",
                                        "outcome": "executed", "failure_reason": "post_commit"},
                                       now=1)
        with self.assertRaisesRegex(CNPError, "proof"):
            self.scheduler.retry_failed_task("fixture", "j1", [bid("r2")], now=1,
                                             lease_duration=2)

    def test_invalid_retry_preserves_previous_assignment(self):
        scheduler = self.scheduler
        scheduler.update_feedback({**self.first, "execution_status": "failed",
                                   "outcome": "not_executed", "failure_reason": "cancelled"},
                                  now=1)
        with self.assertRaises(CNPError):
            scheduler.retry_failed_task("fixture", "j1", [bid("r2", "wrong")], now=1,
                                        lease_duration=2)
        self.assertEqual(scheduler.get_task_status("fixture", "j1"), "failed")
        self.assertEqual(scheduler.get_assignment("fixture", "j1"), self.first)


class TestFreezeCandidateContract(unittest.TestCase):
    def test_main_local_single_worker_releases_lock_and_late_message_cannot_take_new_lock(self):
        scheduler = SchedulingAgent()
        a, b = task("a"), task("b")
        self.assertEqual(scheduler.schedule(a, [bid("one", "a")])["winner_id"], "one")
        self.assertIsNone(scheduler.schedule(b, [bid("one", "b")])["winner_id"])
        feedback = {"task_id": "fixture", "subtask_id": "a", "execution_status": "completed"}
        scheduler.update_feedback(feedback)
        self.assertEqual(scheduler.schedule(b, [bid("one", "b")])["winner_id"], "one")
        scheduler.update_feedback(feedback)
        with self.assertRaises(CNPError):
            scheduler.update_feedback({**feedback, "execution_status": "failed",
                                       "failure_reason": "late_conflict"})
        self.assertEqual(scheduler.active_agent_tasks, {"one": "b"})


if __name__ == "__main__":
    unittest.main()
