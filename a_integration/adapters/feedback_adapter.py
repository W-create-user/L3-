"""Translate B ExecutionResult into C's latest feedback contract."""
from __future__ import annotations


def classify_failure(result):
    status = result.get("execution_status")
    if status == "rejected":
        return "safety_rejected", False
    reason = str(result.get("failure_reason") or "").lower()
    if "timeout" in reason:
        return "execution_timeout", True
    if "agent" in reason or "winner" in reason or "unavailable" in reason:
        return "agent_unavailable", True
    return "environment_error", False


def execution_result_to_feedback(result):
    if not isinstance(result, dict):
        raise TypeError("ExecutionResult must be a dict")
    status = result.get("execution_status")
    if status == "not_assigned":
        return None
    if status == "completed":
        normalized, reason, exclude = "completed", None, False
    elif status in ("failed", "rejected"):
        normalized = "failed"
        failure_type, exclude = classify_failure(result)
        safety = result.get("safety_decision") or {}
        rule = safety.get("rule_id") if isinstance(safety, dict) else None
        reason = failure_type + (f":{rule}" if rule else "")
    elif status == "running":
        normalized, reason, exclude = "running", None, False
    else:
        raise ValueError(f"Unsupported execution_status: {status!r}")
    task_id, subtask_id = result.get("task_id"), result.get("subtask_id")
    if not task_id or not subtask_id:
        raise ValueError("ExecutionResult must preserve task_id and subtask_id")
    return {
        "task_id": task_id,
        "subtask_id": subtask_id,
        "execution_status": normalized,
        "failure_reason": reason,
        "exclude_failed_agent": exclude,
    }


def update_scheduler_from_result(scheduler, result):
    feedback = execution_result_to_feedback(result)
    if feedback is None:
        return {"ignored": True, "reason": "not_assigned"}
    return scheduler.update_feedback(feedback)
