"""A02 draft GoalSpec validation, deliberately not wired to physical execution.

The variable registry is supplied by a trusted, versioned B/D handoff. Matching
metadata does not prove physical calibration, reachability or safety approval.
"""
from copy import deepcopy
import math


SCHEMA_VERSION = "2.0-draft"


def _fields(value, required, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required)
    if missing or unknown:
        raise ValueError(f"{label}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty canonical string")


def _number(value, label):
    # Reject bool and numeric strings, including before serialization.
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite JSON number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite")


def _integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _interval(value, label):
    _fields(value, ("lower", "upper"), label)
    for key in ("lower", "upper"):
        _number(value[key], f"{label}.{key}")
    if value["lower"] > value["upper"]:
        raise ValueError(f"{label} is reversed")


def validate_registry(registry):
    """Validate the proposed normalized registry, not B's raw export format."""
    _fields(registry, ("schema_version", "environment_id", "registry_version",
                       "evidence_ref", "observation_size", "action_size", "variables"), "registry")
    if registry["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported registry schema_version")
    for key in ("environment_id", "registry_version", "evidence_ref"):
        _text(registry[key], key)
    for key in ("observation_size", "action_size"):
        _integer(registry[key], 1, key)
    variables = registry["variables"]
    if not isinstance(variables, dict) or not variables:
        raise ValueError("registry.variables must be a nonempty object")
    seen = set()
    for name, item in variables.items():
        _text(name, "variable name")
        _fields(item, ("source", "index", "unit", "coordinate", "bounds"), name)
        if item["source"] not in ("observation", "action"):
            raise ValueError("variable source must be observation or action")
        _integer(item["index"], 0, f"{name}.index")
        if item["index"] >= registry[item["source"] + "_size"]:
            raise ValueError(f"{name}.index exceeds vector size")
        slot = (item["source"], item["index"])
        if slot in seen:
            raise ValueError("duplicate source/index mapping")
        seen.add(slot)
        for key in ("unit", "coordinate"):
            _text(item[key], f"{name}.{key}")
            if item[key].lower() in ("unknown", "unconfirmed", "pending", "tbd"):
                raise ValueError(f"{name}.{key} is unresolved")
        if item["bounds"] is not None:
            _interval(item["bounds"], f"{name}.bounds")
        if item["source"] == "action" and item["bounds"] is None:
            raise ValueError("action bounds are required")
    expected = set(range(registry["action_size"]))
    actual = {idx for source, idx in seen if source == "action"}
    if actual != expected:
        raise ValueError("registry must describe every action axis")
    return deepcopy(registry)


def validate_goal_spec(goal, registry):
    """Return an isolated, validated draft. Never authorize an action or success.

    Bounds constrain the declared numeric domain, not a chemical safety envelope.
    The full requested action range must be inside B's model-coordinate range.
    No implicit unit conversion, clipping, default target or fallback variable.
    """
    registry = validate_registry(registry)
    _fields(goal, ("schema_version", "goal_id", "environment_id", "registry_version",
                   "variable", "unit", "coordinate", "target_range", "max_steps",
                   "hold_steps", "allowed_actions"), "GoalSpec")
    if goal["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported GoalSpec schema_version")
    _text(goal["goal_id"], "goal_id")
    for key in ("environment_id", "registry_version"):
        if goal[key] != registry[key]:
            raise ValueError(f"GoalSpec {key} mismatch")
    _text(goal["variable"], "variable")
    variable = registry["variables"].get(goal["variable"])
    if variable is None or variable["source"] != "observation":
        raise ValueError("goal requires a mapped observation variable")
    for key in ("unit", "coordinate"):
        if goal[key] != variable[key]:
            raise ValueError(f"GoalSpec {key} mismatch; no implicit conversion")
    _interval(goal["target_range"], "target_range")
    bounds = variable["bounds"]
    if bounds and (goal["target_range"]["lower"] < bounds["lower"]
                   or goal["target_range"]["upper"] > bounds["upper"]):
        raise ValueError("target_range exceeds declared variable domain")
    for key in ("max_steps", "hold_steps"):
        _integer(goal[key], 1, key)
    if goal["hold_steps"] > goal["max_steps"]:
        raise ValueError("hold_steps exceeds max_steps")
    actions = goal["allowed_actions"]
    expected = {name for name, item in registry["variables"].items() if item["source"] == "action"}
    _fields(actions, expected, "allowed_actions")
    for name, action in actions.items():
        _fields(action, ("unit", "coordinate", "range"), name)
        variable = registry["variables"][name]
        for key in ("unit", "coordinate"):
            if action[key] != variable[key]:
                raise ValueError(f"{name}.{key} mismatch")
        _interval(action["range"], f"{name}.range")
        bounds = variable["bounds"]
        if (action["range"]["lower"] < bounds["lower"]
                or action["range"]["upper"] > bounds["upper"]):
            raise ValueError(f"{name} range exceeds model bounds")
    return deepcopy(goal)
