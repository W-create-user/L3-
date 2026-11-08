"""Convert DecisionAgent Task dictionaries into C SchedulingAgent records."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping


def convert_subtasks_for_c(decision_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    required_parent = ("task_id", "priority", "subtasks")
    missing = [key for key in required_parent if key not in decision_result]
    if missing:
        raise ValueError(f"DecisionResult missing fields: {missing}")

    parent_id = decision_result["task_id"]
    priority = decision_result["priority"]
    if not isinstance(parent_id, str) or not parent_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
        raise ValueError("priority must be an integer in [1, 5]")
    if not isinstance(decision_result["subtasks"], list) or not decision_result["subtasks"]:
        raise ValueError("subtasks must be a non-empty list")

    parent_constraints = decision_result.get("constraints", {})
    if not isinstance(parent_constraints, dict):
        raise ValueError("constraints must be a dictionary")

    raw_ids = [item.get("subtask_id") for item in decision_result["subtasks"]]
    if any(not isinstance(x, str) or not x.strip() for x in raw_ids):
        raise ValueError("every subtask must have a non-empty subtask_id")
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("duplicate subtask_id in DecisionResult")
    valid_ids = set(raw_ids)

    converted: list[dict[str, Any]] = []
    for item in decision_result["subtasks"]:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"subtask {item['subtask_id']} has invalid name")
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
            raise ValueError(f"subtask {item['subtask_id']} dependencies must be a list of IDs")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"subtask {item['subtask_id']} has duplicate dependencies")
        unknown = set(dependencies) - valid_ids
        if unknown:
            raise ValueError(f"subtask {item['subtask_id']} has unknown dependencies: {sorted(unknown)}")
        if item["subtask_id"] in dependencies:
            raise ValueError(f"subtask {item['subtask_id']} cannot depend on itself")
        status = item.get("status", "pending")
        if not isinstance(status, str):
            raise ValueError("subtask status must be a string")
        status = status.lower()
        if status not in {"pending", "running", "completed", "failed", "blocked"}:
            raise ValueError(f"unsupported subtask status: {status}")

        record = {
            "task_id": parent_id,
            "subtask_id": item["subtask_id"],
            "task_type": name,
            "priority": priority,
            "dependencies": list(dependencies),
            "status": status,
            "constraints": deepcopy(parent_constraints),
            "required_capability": item.get("required_capability", "decision"),
            "description": item.get("description", ""),
            "order": item.get("order"),
            "preconditions": list(item.get("preconditions", [])),
            "completion_criteria": list(item.get("completion_criteria", [])),
        }
        converted.append(record)
    return converted


def enrich_assignment(assignment: Mapping[str, Any], tasks_for_c: list[dict[str, Any]]) -> dict[str, Any]:
    """Add task_type from A's canonical HTN record; do not invent/alter IDs."""
    required = ("task_id", "subtask_id", "winner_id", "assignment_status")
    missing = [key for key in required if key not in assignment]
    if missing:
        raise ValueError(f"Assignment missing fields: {missing}")
    matches = [t for t in tasks_for_c if t["subtask_id"] == assignment["subtask_id"]]
    if len(matches) != 1:
        raise ValueError("Assignment subtask_id must match exactly one A HTN subtask")
    task = matches[0]
    if assignment["task_id"] != task["task_id"]:
        raise ValueError("Assignment task_id does not match A's parent task")
    result = dict(assignment)
    result["task_type"] = task["task_type"]
    result["constraints"] = deepcopy(task["constraints"])
    result["priority"] = task["priority"]
    return result
