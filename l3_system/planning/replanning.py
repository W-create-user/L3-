"""Bounded recovery policy for A; no implicit infinite retries."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_RESOURCE = "no_resource"
    MESSAGE_TIMEOUT = "message_timeout"
    EXECUTION_UNKNOWN = "execution_result_unknown"
    ENVIRONMENT_ERROR = "environment_error"
    UNKNOWN_STATE = "unknown_state"


def classify_failure(result):
    if not isinstance(result, dict):
        return FailureType.UNKNOWN_STATE
    if result.get('execution_status') == 'completed':
        return None
    reason = result.get('failure_reason')
    # Any possible dispatch takes precedence over a textual rejection/timeout.
    if (result.get('step_called') is True or result.get('executed') is True
            or reason == 'execution_result_unknown'):
        return FailureType.EXECUTION_UNKNOWN
    if result.get('execution_status') == 'rejected' or reason == 'safety_rejected':
        if result.get('executed') is False and result.get('step_called') is False:
            return FailureType.CANDIDATE_REJECTED
        return FailureType.EXECUTION_UNKNOWN
    if reason in ('execution_timeout', 'message_timeout'):
        return (FailureType.MESSAGE_TIMEOUT if result.get('step_called') is False
                and result.get('executed') is False else FailureType.EXECUTION_UNKNOWN)
    if reason in ('no_resource', 'agent_unavailable'):
        return FailureType.NO_RESOURCE
    if reason == 'environment_error':
        return FailureType.ENVIRONMENT_ERROR
    return FailureType.UNKNOWN_STATE


@dataclass
class ReplanningController:
    max_action_regenerations: int = 2
    max_environment_retries: int = 1
    action_regenerations: dict[str, int] = field(default_factory=dict)
    environment_retries: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for value in (self.max_action_regenerations, self.max_environment_retries):
            if type(value) is not int or value < 0:
                raise ValueError('recovery budgets must be nonnegative integers')

    def classify(self, result):
        """Use structured facts; timeout text never proves an action was unexecuted."""
        kind = classify_failure(result)
        result = result if isinstance(result, dict) else {}
        unexecuted = (result.get("executed") is False
                      and result.get("step_called") is False)
        if kind == FailureType.CANDIDATE_REJECTED:
            retryable = unexecuted and self.action_regenerations.get(self._key(result), 0) < self.max_action_regenerations
            action = "regenerate" if retryable else "takeover"
        elif kind in {FailureType.NO_RESOURCE, FailureType.MESSAGE_TIMEOUT}:
            retryable = (unexecuted and result.get("release_confirmed") is True
                         and self.environment_retries.get(self._key(result), 0) < self.max_environment_retries)
            action = "reassign" if retryable else "takeover"
        else:
            retryable = False
            action = "reconcile_then_takeover" if kind == FailureType.EXECUTION_UNKNOWN else "takeover"
        if retryable and any(not isinstance(result.get(key), str) or not result[key].strip()
                             for key in ('task_id', 'subtask_id')):
            retryable, action = False, 'takeover'
        safety = result.get("safety_decision") or {}
        return {"failure_type": kind.value if kind else None, "retryable": retryable,
                "action": action, "rule_id": safety.get("rule_id") if isinstance(safety, dict) else None}

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
        if classification["failure_type"] == FailureType.CANDIDATE_REJECTED.value and classification["retryable"]:
            return {"assignment_status": "action_regeneration_required",
                    "reason": "safety_rejected"}
        if not classification["retryable"]:
            return {"assignment_status": "not_retried",
                    "reason": classification["failure_type"] or "not_retryable",
                    "recovery_action": classification["action"]}
        if scheduler.active_agent_tasks:
            return {"assignment_status": "not_retried", "reason": "release_not_confirmed"}
        if not self.record_environment_retry(result):
            return {"assignment_status": "retry_limit_reached"}
        return scheduler.retry_failed_task(
            result["task_id"], result["subtask_id"], new_bids
        )
