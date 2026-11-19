"""Run explicitly supplied numeric simulation goals through the local A/B/C/D chain."""
from dataclasses import asdict
import math
import time
from uuid import uuid4

from .goals import evaluate_goal, validate_goal_spec, validate_registry


def open_environment(environment, *, source_dir=None, max_steps=60, atropine_factory=None):
    from l3_system.execution import EnvironmentAdapter
    if environment == "AtropineEnvGym":
        return (atropine_factory or EnvironmentAdapter)(source_dir=source_dir, max_steps=max_steps)
    if environment == "MAbEnvGym":
        try:
            from l3_system.execution.mab_adapter import MAbAdapter
        except ModuleNotFoundError as exc:
            if exc.name != "l3_system.execution.mab_adapter":
                raise
            raise RuntimeError("MAbEnvGym requires the B03 adapter in this checkout") from exc
        # B's adapter alone is insufficient: the executor must label its requests
        # with the B03 environment metadata rather than the legacy Atropine fields.
        from l3_system.execution.execution_agent import ExecutionAgent
        if not callable(getattr(ExecutionAgent, "_backend_metadata", None)):
            raise RuntimeError("MAbEnvGym requires B03 environment-aware ActionRequests")
        return MAbAdapter(source_dir=source_dir, max_steps=max_steps)
    raise ValueError(f"Unknown environment: {environment}")


class GoalCycle:
    def __init__(self, plan, sample, goal, registry, adapter, state_version, deadline):
        self.plan = plan
        self.observation = sample["observation"]
        self.target_range = goal["target_range"]
        self.adapter = adapter
        self.state_version = state_version
        self.deadline = deadline
        self.complete_ids = set()
        self.facts = {"observation_valid": sample["safety_ok"],
                      "goal_achieved": plan.evaluation["achieved"]}
        self.action_limits = sorted(
            (variable["index"], goal["allowed_actions"][name]["range"])
            for name, variable in registry["variables"].items() if variable["source"] == "action"
        )

    def parent_task(self):
        def subtask_id(name):
            return f"{self.plan.plan_id}:{name}"

        return {
            "task_id": self.plan.plan_id, "priority": 3, "task_type": "control_cycle",
            "description": f"Numeric simulation goal {self.plan.goal_id}",
            "status": "created", "constraints": {},
            "subtasks": [
                {"subtask_id": subtask_id(task.task_id), "name": task.task_id,
                 "order": order, "required_capability": task.capability,
                 "dependencies": [subtask_id(dep) for dep in task.dependencies],
                 "preconditions": list(task.preconditions),
                 "completion_criteria": list(task.completion_criteria)}
                for order, task in enumerate(self.plan.tasks, 1)
            ],
        }

    def require_ready(self, name):
        if name in {"evaluate_goal", "verify_goal"}:
            self.require_dispatch()
        if not any(task.task_id == name for task in self.plan.ready_tasks(self.facts, self.complete_ids)):
            raise RuntimeError(f"HTN prerequisites unsatisfied: {name}")

    def complete(self, name):
        task = next(task for task in self.plan.tasks if task.task_id == name)
        facts = {**self.facts, **dict.fromkeys(task.completion_criteria, True)}
        self.facts.update(self.plan.verify_completion(name, facts, self.complete_ids))
        self.complete_ids.add(name)

    def check_action(self, action):
        for index, limits in self.action_limits:
            if not limits["lower"] <= action[index] <= limits["upper"]:
                raise ValueError(f"Candidate outside goal action range at index {index}")

    def require_dispatch(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Goal deadline reached before dispatch")
        if self.adapter.state_version != self.state_version:
            raise RuntimeError("Observation changed since goal planning")


def run_numeric_goal(system, goal, registry, *, environment, timeout_seconds=None, policy=None):
    registry = validate_registry(registry)
    goal = validate_goal_spec(goal, registry)
    adapter = system.executor.adapter
    if (environment != registry["environment_id"]
            or adapter.observation_space.shape != (registry["observation_size"],)
            or adapter.action_space.shape != (registry["action_size"],)):
        raise ValueError("Live environment does not match the goal registry")
    metadata = getattr(adapter, "metadata", None)
    if metadata is not None:
        if metadata.get("environment_id") != environment:
            raise ValueError("B adapter environment does not match the goal registry")
        for variable in registry["variables"].values():
            coordinate = metadata.get(f'{variable["source"]}_coordinate')
            if variable["coordinate"] != coordinate:
                raise ValueError("B adapter coordinates differ from the registry; explicit projection is required")
    if timeout_seconds is not None and (
        type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("Goal timeout must be finite and positive")
    if system.state != "autonomous" or adapter.step_count != 0:
        raise RuntimeError("Numeric goal requires an autonomous session at a fresh reset")
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    episode = str(uuid4())
    cycles, samples, plans = [], [], []

    def sample(safety_ok):
        return {"goal_id": goal["goal_id"], "environment_id": environment,
                "registry_version": registry["registry_version"], "episode_id": episode,
                "step": adapter.step_count, "observation": system.executor.get_observation(),
                "safety_ok": safety_ok}

    version = adapter.state_version
    initial_sample = sample(False)
    initial = system.monitor.check_observation(initial_sample["observation"], goal["goal_id"])
    initial_sample["safety_ok"] = initial.get("approved") is True
    samples.append(initial_sample)
    while True:
        evaluation = evaluate_goal(samples[-1], samples[:-1], goal, registry)
        if deadline is not None and time.monotonic() >= deadline:
            evaluation.update(status="failed", achieved=False, reason="goal_deadline_exceeded")
            break
        if evaluation["status"] in {"failed", "unevaluable"}:
            if evaluation["reason"] == "safety_violation":
                system._transition("takeover_requested", "numeric_goal_observation_rejected", task_id=goal["goal_id"])
            break
        if adapter.done and not evaluation["achieved"]:
            evaluation.update(status="failed", achieved=False, reason="episode_ended_before_goal")
            break
        plan = system.decision.create_goal_plan(
            goal, registry, samples[-1], samples[:-1],
            capabilities={"decision", "monitoring", "environment_execution"},
            parent_plan_id=plans[-1]["plan_id"] if plans else None,
        )
        plans.append(asdict(plan))
        driver = GoalCycle(plan, samples[-1], goal, registry, adapter, version, deadline)
        result = system.run_cycle(action_generator=policy, goal_cycle=driver)
        cycles.append(result)
        if result["summary"]["run_status"] != "completed":
            evaluation.update(status="failed", achieved=False, reason=result["summary"]["failure_reason"])
            break
        if evaluation["achieved"]:
            break
        execution = next(stage["execution_result"] for stage in result["stage_results"]
                         if stage["task_type"] == "execute_approved_action")
        version = execution["state_version"]
        samples.append({**samples[-1], "step": execution["step_count"],
                        "observation": execution["observation"], "safety_ok": True})

    return {"summary": {**evaluation, "goal_id": goal["goal_id"], "environment": environment,
                        "step_count": adapter.step_count, "system_state": system.state,
                        "r03_enabled": False, "production_acceptance": False},
            "goal_spec": goal, "registry": registry, "episode_id": episode,
            "plans": plans, "samples": samples, "cycles": cycles}
