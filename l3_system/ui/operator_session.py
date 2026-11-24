"""A's local operator workflow around the existing simulation runtime."""
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from l3_system.planning.goal_runtime import run_numeric_goal
from l3_system.planning.goals import validate_goal_spec, validate_registry


class OperatorSession:
    def __init__(self, system, environment, goal=None, registry=None, *, separate_confirmation=False):
        if (goal is None) != (registry is None):
            raise ValueError("Supply both a GoalSpec and its registry")
        self.system = system
        self.separate_confirmation = separate_confirmation
        self.environment = environment
        self.registry = validate_registry(registry) if registry is not None else None
        self.goal = validate_goal_spec(goal, self.registry) if goal is not None else None
        if self.goal is not None and self.goal["environment_id"] != environment:
            raise ValueError("GoalSpec does not belong to this environment")
        self.acknowledged = None
        self.last_goal_result = None
        self.episode_history_start = 0
        self.updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def adapter(self):
        return self.system.executor.adapter

    def context(self):
        last = self.system.transitions[-1] if self.system.transitions else {}
        task_id = last.get("task_id")
        if task_id is None and self.system.history:
            task_id = self.system.history[-1]["summary"]["task_id"]
        return {"task_id": task_id, "state_version": self.adapter.state_version}

    def record(self, reason, operator, note, **extra):
        self.system._transition(self.system.state, reason, operator=operator, note=note,
                                **self.context(), **extra)

    @staticmethod
    def require_operator(operator, reason):
        if any(not isinstance(value, str) or not value.strip() for value in (operator, reason)):
            raise ValueError("Operator and reason are required")

    def pause(self, operator, reason):
        self.require_operator(operator, reason)
        if self.system.state != "autonomous":
            raise RuntimeError("Session is already restricted")
        self.acknowledged = None
        self.system._transition("takeover_requested", "operator_requested_pause",
                                operator=operator, note=reason, **self.context())

    def require_takeover(self):
        if self.system.state != "takeover_requested":
            raise RuntimeError("Takeover has not been requested")

    def confirmation_key(self):
        return (len(self.system.transitions), self.adapter.state_version)

    def confirm(self, operator, reason):
        self.require_operator(operator, reason)
        self.require_takeover()
        self.record("operator_confirmed_takeover", operator, reason)
        self.acknowledged = self.confirmation_key()

    def revise_goal(self, operator, reason, *, lower, upper):
        self.require_operator(operator, reason)
        self.require_takeover()
        if self.goal is None:
            raise RuntimeError("Start the console with a GoalSpec to revise its target")
        candidate = deepcopy(self.goal)
        candidate["goal_id"] = str(uuid4())
        candidate["target_range"] = {"lower": float(lower), "upper": float(upper)}
        candidate = validate_goal_spec(candidate, self.registry)
        self.record("operator_revised_goal", operator, reason,
                    previous_goal_id=self.goal["goal_id"], revised_goal=candidate)
        self.goal = candidate
        self.last_goal_result = None
        self.acknowledged = None

    def resume(self, operator, reason, *, reset=False):
        self.require_operator(operator, reason)
        self.require_takeover()
        if self.separate_confirmation and self.acknowledged != self.confirmation_key():
            raise RuntimeError("Confirm takeover against the current simulation state")
        if self.goal is not None and self.adapter.step_count != 0 and not reset:
            raise RuntimeError("A numeric goal requires a fresh episode")
        if self.system.history and not reset:
            failure = self.system.history[-1]["summary"].get("failure_classification", {})
            if failure.get("failure_type") == "execution_result_unknown":
                raise RuntimeError("An uncertain execution requires reconciliation or a reviewed new episode")
        context = self.context()
        self.system.resume(operator=operator, reason=reason, reset=reset)
        self.acknowledged = None
        # Record the web operation separately with its original task association.
        self.system._transition(self.system.state, "operator_resumed_console",
                                operator=operator, note=reason, **context,
                                resumed_state_version=self.adapter.state_version)
        if reset:
            self.episode_history_start = len(self.system.history)
            self.last_goal_result = None
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def cycle(self):
        if self.goal is None:
            result = self.system.run_cycle()
        else:
            result = run_numeric_goal(self.system, self.goal, self.registry,
                                      environment=self.environment, timeout_seconds=60)
            self.last_goal_result = result
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return result
