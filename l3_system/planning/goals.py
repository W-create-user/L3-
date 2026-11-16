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


def evaluate_goal(observation, history, goal, registry):
    """Evaluate ordered, post-step samples; reset (step=0) never counts as hold.

    Samples bind environment/registry/episode and carry the caller's explicit
    safety_ok verdict. This is a numeric contract, not process-safety certification.
    History is the complete earlier sample sequence for this goal, not rewards.
    Invalid samples yield unevaluable, never success. Invalid specs raise.
    """
    goal = validate_goal_spec(goal, registry)
    result = dict(status='unevaluable', achieved=False, in_range=False,
                  deviation=None, consecutive_steps=0, remaining_steps=None,
                  reason=None)
    try:
        if not isinstance(history, list):
            raise ValueError('history must be a list')
        samples = history + [observation]
        previous = None
        episode = None
        count = 0
        safe = True
        variable = registry['variables'][goal['variable']]
        low, high = goal['target_range']['lower'], goal['target_range']['upper']
        for sample in samples:
            _fields(sample, ('goal_id', 'environment_id', 'registry_version', 'episode_id',
                             'step', 'observation', 'safety_ok'), 'sample')
            for key in ('goal_id', 'environment_id', 'registry_version'):
                if sample[key] != goal[key]:
                    raise ValueError(f'sample {key} mismatch')
            _text(sample['episode_id'], 'episode_id')
            if episode is not None and sample['episode_id'] != episode:
                raise ValueError('mixed episodes')
            episode = sample['episode_id']
            step = sample['step']
            _integer(step, 0, 'step')
            if (previous is None and step not in (0, 1)) or (previous is not None and step != previous + 1):
                raise ValueError('history must be contiguous from step 0 or 1; no duplicate samples')
            previous = step
            if type(sample['safety_ok']) is not bool:
                raise ValueError('explicit boolean safety_ok required')
            safe = safe and sample['safety_ok']
            vector = sample['observation']
            if not isinstance(vector, (list, tuple)) or len(vector) != registry['observation_size']:
                raise ValueError('observation shape mismatch')
            for value in vector:
                _number(value, 'observation value')
            value = vector[variable['index']]
            bounds = variable['bounds']
            if bounds and not bounds['lower'] <= value <= bounds['upper']:
                raise ValueError('goal variable outside declared domain')
            inside = low <= value <= high
            count = count + 1 if step > 0 and inside else 0
        result.update(in_range=inside, deviation=value-low if value < low else value-high if value > high else 0,
                      consecutive_steps=count, remaining_steps=max(0, goal['max_steps']-step))
        if not safe:
            result.update(status='failed', reason='safety_violation')
        elif step > goal['max_steps']:
            result.update(status='failed', reason='step_budget_exceeded')
        elif count >= goal['hold_steps']:
            result.update(status='succeeded', achieved=True)
        elif step == goal['max_steps']:
            result.update(status='failed', reason='goal_not_met_by_deadline')
        else:
            result.update(status='pending', reason='hold_window_incomplete' if inside else 'outside_target')
    except ValueError as exc:
        result.update(reason=str(exc))
    return result
