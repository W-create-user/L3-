"""Fail-closed adapter from B ActionRequest to D MonitoringAgent v1.2."""
from __future__ import annotations
from collections.abc import Mapping


def make_d_safety_checker(monitor):
    if getattr(monitor, "demo_mode", True):
        raise RuntimeError("Formal integration refuses D demo_mode; initialize with real adapter.action_space")

    def check(action_request, observation=None):
        if not isinstance(action_request, Mapping):
            raise TypeError("action_request must be a mapping")
        task_id = action_request.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        if "action" not in action_request:
            raise ValueError("action is required")
        raw = monitor.check_request(dict(action_request), observation)
        if not isinstance(raw, dict):
            raise TypeError("D check_request must return a dict")
        if raw.get("task_id") != task_id:
            raise ValueError("D task_id mismatch")
        decision = raw.get("decision")
        if decision not in ("pass", "reject"):
            raise ValueError("D decision must be exactly pass or reject")
        if raw.get("demo_mode") is True:
            raise ValueError("D returned a demo safety result")
        if "approved" in raw and (type(raw["approved"]) is not bool or raw["approved"] != (decision == "pass")):
            raise ValueError("D approved conflicts with decision")
        if "status" in raw and str(raw["status"]).lower() != decision:
            raise ValueError("D status conflicts with decision")
        if raw.get("request_id") is not None and raw["request_id"] != action_request.get("request_id"):
            raise ValueError("D request_id mismatch")
        if raw.get("subtask_id") is not None and raw["subtask_id"] != action_request.get("subtask_id"):
            raise ValueError("D subtask_id mismatch")
        return {
            "task_id": task_id,
            "decision": decision,
            "request_id": action_request.get("request_id"),
            "subtask_id": action_request.get("subtask_id"),
            "rule_id": raw.get("rule_id"),
            "reason": raw.get("reason"),
            "risk_level": raw.get("risk_level"),
            "demo_mode": bool(getattr(monitor, "demo_mode", True)),
            "raw_safety_result": raw,
        }
    return check
