"""Validate C Assignment against A's original HTN task before dispatch to B."""
from __future__ import annotations
from copy import deepcopy


def build_b_task(assignment, parent_task, subtask):
    required = ("task_id", "subtask_id", "winner_id", "assignment_status")
    missing = [k for k in required if k not in assignment]
    if missing:
        raise ValueError(f"Assignment missing fields: {missing}")
    if assignment["assignment_status"] != "awarded":
        raise ValueError("Only awarded Assignment can be dispatched")
    if assignment["task_id"] != parent_task.get("task_id"):
        raise ValueError("Assignment parent task_id mismatch")
    if assignment["subtask_id"] != subtask.get("subtask_id"):
        raise ValueError("Assignment subtask_id mismatch")
    if not assignment["winner_id"]:
        raise ValueError("Assignment winner_id is empty")
    # B normalizes this Task contract. Preserve original HTN structure for prepare_action.
    task = deepcopy(parent_task)
    task["task_name"] = task.get("description", task.get("task_name", "L3 task"))
    params = dict(task.get("parameters", {}))
    params.setdefault("task_type", task.get("task_type", "control_cycle"))
    params.setdefault("subtasks", deepcopy(parent_task.get("subtasks", [])))
    task["parameters"] = params
    return task


def assert_winner(assignment, executor_id):
    if assignment.get("assignment_status") != "awarded":
        raise ValueError("Assignment is not awarded")
    if not isinstance(executor_id, str) or not executor_id:
        raise ValueError("executor_id must be configured")
    if assignment.get("winner_id") != executor_id:
        raise PermissionError(
            f"Assignment winner {assignment.get('winner_id')!r} does not match executor {executor_id!r}"
        )
