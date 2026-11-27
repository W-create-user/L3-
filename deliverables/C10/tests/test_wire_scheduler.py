import unittest
from unittest.mock import patch

from deliverables.C10.live import SchedulingService
from deliverables.C08.tests.test_c_delivery import bid, task


class WireSchedulerTests(unittest.TestCase):
    def test_feedback_requires_original_award_and_cannot_release_new_lock(self):
        scheduler = SchedulingService(10)
        first = scheduler.schedule(task("a"), [bid("one", "a")])
        receipt = {"task_id": "fixture", "subtask_id": "a", "execution_status": "completed"}
        with self.assertRaisesRegex(ValueError, "assignment_id"):
            scheduler.update_feedback(receipt, "wrong")
        scheduler.update_feedback(receipt, first["assignment_id"])
        second = scheduler.schedule(task("b"), [bid("one", "b")])
        self.assertNotEqual(first["assignment_id"], second["assignment_id"])
        scheduler.update_feedback(receipt, first["assignment_id"])
        with self.assertRaisesRegex(ValueError, "assignment_id"):
            scheduler.update_feedback({**receipt, "subtask_id": "b"}, first["assignment_id"])
        self.assertEqual(scheduler.locks(), {"one": "b"})

    def test_expired_lease_blocks_dispatch_and_preserves_worker_lock(self):
        scheduler = SchedulingService(10)
        with patch("deliverables.C10.live.monotonic", return_value=0):
            award = scheduler.schedule(task(), [bid("one")])
        with patch("deliverables.C10.live.monotonic", return_value=10):
            with self.assertRaises(TimeoutError):
                scheduler.assignment_for_execution(award)
            with self.assertRaises(TimeoutError):
                scheduler.update_feedback({"task_id": "fixture", "subtask_id": "j1",
                                           "execution_status": "completed"}, award["assignment_id"])
        self.assertEqual(scheduler.locks(), {"one": "j1"})


if __name__ == "__main__":
    unittest.main()
