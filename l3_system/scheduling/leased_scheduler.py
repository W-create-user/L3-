"""Opt-in CNP scheduling with correlated terminal feedback."""
from dataclasses import dataclass
import math
from uuid import uuid4

from l3_system.agents.scheduling_agent import SchedulingAgent
from l3_system.algorithms.cnp_protocol import CNPError


@dataclass(frozen=True)
class Lease:
    assignment_id: str
    attempt: int
    deadline: float


class LeasedScheduler(SchedulingAgent):
    """Single coordinator owns scheduling state and all reconciliation decisions.

    An expired lease keeps its worker reserved. Retry requires a trusted
    acknowledgement from the execution owner that the original work did not run.
    """

    def __init__(self, cnp=None, *, assignment_namespace=None):
        super().__init__(cnp)
        self.namespace = assignment_namespace or uuid4().hex
        self.leases = {}
        self.expired = set()
        self.retryable = set()
        self.receipts = {}

    def schedule(self, task, bids, *, now, lease_duration):
        deadline = now + lease_duration
        if not math.isfinite(now) or not math.isfinite(deadline) or deadline <= now:
            raise CNPError("Lease duration and clock must produce a finite future deadline")
        # Validate even bids that would subsequently be excluded by resource state.
        self.cnp.collect_bids(task, bids := list(bids))
        result = super().schedule(task, bids)
        if result["assignment_status"] == "awarded":
            key = self._key(task)
            attempt = len(self.assignment_history[key])
            lease = Lease(f"{self.namespace}:{key}:{attempt}", attempt, deadline)
            self.leases[key] = lease
            identity = {"assignment_id": lease.assignment_id,
                        "attempt": lease.attempt, "lease_deadline": lease.deadline}
            self.assignments[key].update(identity)
            self.assignment_history[key][-1].update(identity)
            result.update(identity)
        return result

    def schedule_tasks(self, tasks, bids_by_subtask, *, now, lease_duration):
        self.register_tasks(tasks)
        return [
            self.schedule(task, bids_by_subtask.get(self._key(task), ()),
                          now=now, lease_duration=lease_duration)
            for task in self.ready_tasks()
        ]

    def assignment_for_execution(self, assignment, *, now):
        self.expire(now)
        result = super().assignment_for_execution(assignment)
        active = self.get_active_assignment(assignment["task_id"], assignment.get("subtask_id"))
        if not active or any(assignment.get(k) != v for k, v in active.items()):
            raise CNPError("Assignment is inactive, expired or mismatched")
        return {**result, **{k: active[k] for k in ("assignment_id", "attempt", "lease_deadline")}}

    def get_active_assignment(self, task_id, subtask_id=None):
        assignment = super().get_active_assignment(task_id, subtask_id)
        if assignment is not None and assignment["assignment_id"] in self.expired:
            return None
        return assignment

    def expire(self, now):
        if not math.isfinite(now):
            raise CNPError("Invalid lease clock")
        expired = []
        for key, lease in self.leases.items():
            if (now >= lease.deadline and lease.assignment_id not in self.expired
                    and lease.assignment_id not in self.receipts):
                self.expired.add(lease.assignment_id)
                expired.append({"task_key": key, "assignment_id": lease.assignment_id})
        return expired

    def update_feedback(self, feedback, *, now, reconciled=False):
        self.expire(now)
        key = self._key(feedback)
        lease = self.leases.get(key)
        current = self.assignments.get(key)
        if (lease is None or current is None or feedback.get("task_id") != current["task_id"]
                or feedback.get("assignment_id") != lease.assignment_id
                or type(feedback.get("attempt")) is not int or feedback["attempt"] != lease.attempt
                or feedback.get("winner_id") != current["winner_id"]):
            raise CNPError("Feedback does not match the current assignment")
        status = feedback.get("execution_status")
        outcome = feedback.get("outcome")
        if (status, outcome) not in {("completed", "executed"), ("failed", "not_executed"),
                                      ("failed", "executed")}:
            raise CNPError("Terminal feedback requires a known execution outcome")
        previous = self.receipts.get(lease.assignment_id)
        if previous is not None:
            if previous[0] != feedback:
                raise CNPError("Conflicting terminal receipt")
            return dict(previous[1])
        if lease.assignment_id in self.expired and not reconciled:
            raise CNPError("Expired assignment requires authoritative reconciliation")
        result = super().update_feedback(feedback)
        self.receipts[lease.assignment_id] = (dict(feedback), dict(result))
        if outcome == "not_executed":
            self.retryable.add(key)
        return result

    def retry_failed_task(self, task_id, subtask_id, bids, *, now, lease_duration):
        key = str(subtask_id)
        if key not in self.retryable or self.get_task_status(task_id, subtask_id) != "failed":
            raise CNPError("Retry requires proof the original work did not execute")
        record = self.tasks[key]
        previous = self.assignments.pop(key)
        record["status"] = "pending"
        try:
            result = self.schedule(record, bids, now=now, lease_duration=lease_duration)
        finally:
            if record["status"] != "awarded":
                record["status"] = "failed"
                self.assignments[key] = previous
        if result["assignment_status"] == "awarded":
            self.retryable.remove(key)
        self.retry_log.append({"task_id": task_id, "subtask_id": key,
                               "previous_winner": previous["winner_id"],
                               "new_winner": result["winner_id"],
                               "assignment_status": result["assignment_status"]})
        return result
