"""Bounded recovery policy for A; no implicit infinite retries."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ReplanningController:
    max_action_regenerations: int = 2
    max_environment_retries: int = 1
    action_regenerations: dict[str, int] = field(default_factory=dict)
    environment_retries: dict[str, int] = field(default_factory=dict)

    def classify(self, result):
        if result.get("execution_status") == "rejected":
            safety = result.get("safety_decision") or {}
            rule = safety.get("rule_id") if isinstance(safety, dict) else None
            return {"failure_type": "safety_rejected", "rule_id": rule,
                    "retryable": self.action_regenerations.get(self._key(result), 0) < self.max_action_regenerations}
        if result.get("execution_status") == "failed":
            reason = str(result.get("failure_reason") or "").lower()
            if "timeout" in reason:
                kind = "execution_timeout"
            elif "agent" in reason or "winner" in reason:
                kind = "agent_unavailable"
            else:
                kind = "environment_error"
            key = self._key(result)
            return {"failure_type": kind,
                    "retryable": self.environment_retries.get(key, 0) < self.max_environment_retries}
        return {"failure_type": None, "retryable": False}

    @staticmethod
    def _key(result):
        return f"{result.get('task_id')}::{result.get('subtask_id')}"

    def record_action_regeneration(self, result):
        key = self._key(result)
        count = self.action_regenerations.get(key, 0)
        if count >= self.max_action_regenerations:
            return False
        self.action_regenerations[key] = count + 1
        return True

    def record_environment_retry(self, result):
        key = self._key(result)
        count = self.environment_retries.get(key, 0)
        if count >= self.max_environment_retries:
            return False
        self.environment_retries[key] = count + 1
        return True

    def retry_assignment(self, scheduler, result, new_bids):
        """Request C re-award only for retryable failed execution; no fabricated bids."""
        classification = self.classify(result)
        if not classification["retryable"]:
            return {"assignment_status": "not_retried",
                    "reason": classification["failure_type"] or "not_retryable"}
        if classification["failure_type"] == "safety_rejected":
            # A safety rejection does not prove the agent is faulty. Re-generate/review
            # an action first; do not automatically reassign the same task.
            return {"assignment_status": "action_regeneration_required",
                    "reason": "safety_rejected"}
        if not self.record_environment_retry(result):
            return {"assignment_status": "retry_limit_reached"}
        return scheduler.retry_failed_task(
            result["task_id"], result["subtask_id"], new_bids
        )
